from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Optional, List, TypeVar
import random
import re
import traceback
import copy
import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionInpaintPipeline)
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from modules import devices, script_callbacks, shared
from modules.images import resize_image
from modules.processing import create_infotext, process_images
from modules.sd_models import get_closet_checkpoint_match
from modules.sd_samplers import samplers_for_img2img
from PIL import Image, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo
from torch.hub import download_url_to_file
from torchvision import transforms
from tqdm import tqdm
import os
import gc

import json
import pickle
from abc import ABCMeta, abstractmethod


# ===========================
# Rigister handler
# ===========================



from ia_file_manager import IAFileManager, download_model_from_hf, ia_file_manager
from ia_config import IAConfig, get_ia_config_index, set_ia_config, setup_ia_config_ini
from ia_get_dataset_colormap import create_pascal_label_colormap
from ia_sam_manager import get_sam_mask_generator
from ia_threading import (async_post_reload_model_weights, await_backup_reload_ckpt_info,
                          await_pre_reload_model_weights, clear_cache_decorator,
                          offload_reload_decorator)
from ia_ui_items import (get_cleaner_model_ids, get_inp_model_ids, get_padding_mode_names,
                         get_sam_model_ids, get_sampler_names)
from ia_logging import ia_logging

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
import pickle

T = TypeVar('T')
sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)
def inpaint_anything_api(_: gr.Blocks, app: FastAPI):
    class RespResult(BaseModel):
        code: int = 0
        msg: str = ''
        data: Optional[T]
        @classmethod
        def failed(cls, msg: str = 'failed'):
            return RespResult(code=-1, msg=msg)
        @classmethod
        def success(cls, data = None):
            return RespResult(code=0, msg='Success', data=data)
    @app.get("/inpaint-anything/heartbeat")
    async def heartbeat():
        return RespResult.success()
    class SamPredictRequest(BaseModel):
        image_id: int
        input_image: str
        sam_model_name: str = "sam_vit_h_4b8939.pth"
        anime_style_chk: bool=False

    class SamPredictResp(BaseModel):
        segimg: str = ''
        # saminfo: Optional[Any] = None
    @app.post("/inpaint-anything/sam/image")
    async def run_sam(payload: SamPredictRequest = Body(...)) -> Any:
        print(f"inpaint-anything API /inpaint-anything/sam/image received request")

        global sam_dict
        sam_model_id = payload.sam_model_name
        input_image = decode_to_ndarray(payload.input_image)
        anime_style_chk = payload.anime_style_chk
        sam_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
        if not os.path.isfile(sam_checkpoint):
            return RespResult.failed(f"{sam_model_id} not found, please download")
        if input_image is None:
            return RespResult.failed("Input image not found")

        set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, sam_model_id, IAConfig.SECTIONS.USER)

        if sam_dict["sam_masks"] is not None:
            sam_dict["sam_masks"] = None
            gc.collect()

        ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")

        cm_pascal = create_pascal_label_colormap()
        seg_colormap = cm_pascal
        seg_colormap = np.array([c for c in seg_colormap if max(c) >= 64], dtype=np.uint8)

        sam_mask_generator = get_sam_mask_generator(sam_checkpoint, anime_style_chk)
        ia_logging.info(f"{sam_mask_generator.__class__.__name__} {sam_model_id}")
        try:
            sam_masks = sam_mask_generator.generate(input_image)
        except Exception as e:
            print(traceback.format_exc())
            ia_logging.error(str(e))
            del sam_mask_generator
            return RespResult.failed("SAM generate failed")

        if anime_style_chk:
            for sam_mask in sam_masks:
                sam_mask_seg = sam_mask["segmentation"]
                sam_mask_seg = cv2.morphologyEx(sam_mask_seg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                sam_mask_seg = cv2.morphologyEx(sam_mask_seg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                sam_mask["segmentation"] = sam_mask_seg.astype(bool)

        ia_logging.info("sam_masks: {}".format(len(sam_masks)))
        sam_masks = sorted(sam_masks, key=lambda x: np.sum(x.get("segmentation").astype(np.uint32)))
        if sam_dict["pad_mask"] is not None:
            if (len(sam_masks) > 0 and
                    sam_masks[0]["segmentation"].shape == sam_dict["pad_mask"]["segmentation"].shape and
                    np.any(sam_dict["pad_mask"]["segmentation"])):
                sam_masks.insert(0, sam_dict["pad_mask"])
                ia_logging.info("insert pad_mask to sam_masks")
        sam_masks = sam_masks[:len(seg_colormap)]

        with tqdm(total=len(sam_masks), desc="Processing segments") as progress_bar:
            canvas_image = np.zeros((*input_image.shape[:2], 1), dtype=np.uint8)
            for idx, seg_dict in enumerate(sam_masks[0:min(255, len(sam_masks))]):
                seg_mask = np.expand_dims(seg_dict["segmentation"].astype(np.uint8), axis=-1)
                canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
                seg_color = np.array([idx+1], dtype=np.uint8) * seg_mask * canvas_mask
                canvas_image = canvas_image + seg_color
                progress_bar.update(1)
            seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
            temp_canvas_image = np.apply_along_axis(lambda x: seg_colormap[x[0]], axis=-1, arr=canvas_image)
            if len(sam_masks) > 255:
                canvas_image = canvas_image.astype(bool).astype(np.uint8)
                for idx, seg_dict in enumerate(sam_masks[255:min(509, len(sam_masks))]):
                    seg_mask = np.expand_dims(seg_dict["segmentation"].astype(np.uint8), axis=-1)
                    canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
                    seg_color = np.array([idx+2], dtype=np.uint8) * seg_mask * canvas_mask
                    canvas_image = canvas_image + seg_color
                    progress_bar.update(1)
                seg_colormap = seg_colormap[256:]
                seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
                seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
                canvas_image = np.apply_along_axis(lambda x: seg_colormap[x[0]], axis=-1, arr=canvas_image)
                canvas_image = temp_canvas_image + canvas_image
            else:
                canvas_image = temp_canvas_image
        seg_image = canvas_image.astype(np.uint8)
        seg_img = Image.fromarray(seg_image)
        sam_dict["sam_masks"] = copy.deepcopy(sam_masks)
        # print(sam_dict["sam_masks"])
        del sam_masks
        return RespResult.success(data=SamPredictResp(segimg=encode_to_base64(seg_img)))
    def save_mask(sam_masks):
        filepath = 'output'

    class SamSelectMaskRequest(BaseModel):
        image_id: int
        input_image: str
        select_points: list
        anime_style_chk: bool=False
        expand_mask: Optional[int] = 0
    class SamMaskResp(BaseModel):
        mask: str = ''
        image: str = ''
    @app.post("/inpaint-anything/sam/task")
    async def select_mask(payload: SamSelectMaskRequest = Body(...)) -> Any:
        ignore_black_chk = False
        global sam_dict
        # if sam_dict["sam_masks"] is None:
        #     ia_logging.info("SAM select task failed, sam_dict[\"sam_masks\"] is None")
        #     return RespResult.failed("SAM select task failed")
            # return ret_sel_mask
        sam_masks = sam_dict["sam_masks"]
        input_image = decode_to_ndarray(payload.input_image)
        image = decode_to_ndarray(payload.input_image)
        mask = np.zeros(image.shape[:2] + (1,), dtype=np.uint8)
        selected_mask = np.zeros((*image.shape[:2], 1), dtype=bool)
        selected_points = np.array(payload.select_points)

        # 将选定点的掩码设置为 True
        selected_mask[selected_points[:, 0], selected_points[:, 1]] = True

        # 使用掩码将选定的元素替换为255
        mask[selected_mask] = 255
        
        # mask = sam_image["mask"][:, :, 0:1]

        # if len(sam_masks) > 0 and sam_masks[0]["segmentation"].shape[:2] != mask.shape[:2]:
        #     ia_logging.error("sam_masks shape not match")
        #     ret_sel_mask = None if sel_mask is None else gr.update()
        #     return ret_sel_mask

        canvas_image = np.zeros(image.shape[:2] + (1,), dtype=np.uint8)
        mask_region = np.zeros(image.shape[:2] + (1,), dtype=np.uint8)
        for idx, seg_dict in enumerate(sam_masks):
            seg_mask = np.expand_dims(seg_dict["segmentation"].astype(np.uint8), axis=-1)
            canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
            if (seg_mask * canvas_mask * mask).astype(bool).any():
                mask_region = mask_region + (seg_mask * canvas_mask)
            seg_color = seg_mask * canvas_mask
            canvas_image = canvas_image + seg_color

        if not ignore_black_chk:
            canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
            if (canvas_mask * mask).astype(bool).any():
                mask_region = mask_region + (canvas_mask)

        mask_region = np.tile(mask_region * 255, (1, 1, 3))

        seg_image = mask_region.astype(np.uint8)

        # if invert_chk:
        #     seg_image = np.logical_not(seg_image.astype(bool)).astype(np.uint8) * 255

        sam_dict["mask_image"] = seg_image

        # if image is not None and image.shape == seg_image.shape:
        #     ret_image = cv2.addWeighted(image, 0.5, seg_image, 0.5, 0)
        # else:
        #     ret_image = seg_image
        
        # expand_mask
        if payload.expand_mask > 0:
            seg_image = expand_mask(sam_dict, payload.expand_mask)
            # if input_image is not None and input_image.shape == new_sel_mask.shape:
            #     ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
            # else:
            #     ret_image = new_sel_mask
        
        # run_get_mask
        mask_image_base64 = encode_to_base64(seg_image)

        # run_get_alpha_image
        
        alpha_image = Image.fromarray(input_image).convert("RGBA")
        mask_image = Image.fromarray(seg_image).convert("L")

        alpha_image.putalpha(mask_image)


        return RespResult.success(data=SamMaskResp(mask=mask_image_base64, image=encode_to_base64(alpha_image)))

    def expand_mask(sam_dict, expand_iteration=1):
        # expand_mask
        if sam_dict["mask_image"] is None:
            return None

        new_sel_mask = sam_dict["mask_image"]

        expand_iteration = int(np.clip(expand_iteration, 1, 5))

        new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

        sam_dict["mask_image"] = new_sel_mask

        return new_sel_mask

    def run_get_mask(sam_dict, sel_mask):
        # global sam_dict
        if sam_dict["mask_image"] is None:
            return None

        mask_image = sam_dict["mask_image"]

        save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)

        return mask_image
    def run_get_alpha_image(input_image, sel_mask):
        global sam_dict
        if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
            return None, ""

        mask_image = sam_dict["mask_image"]
        if input_image.shape != mask_image.shape:
            ia_logging.error("The size of image and mask do not match")
            return None, ""

        alpha_image = Image.fromarray(input_image).convert("RGBA")
        mask_image = Image.fromarray(mask_image).convert("L")

        alpha_image.putalpha(mask_image)

        save_name = "_".join([ia_file_manager.savename_prefix, "rgba_image"]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        alpha_image.save(save_name)

        return alpha_image, f"saved: {save_name}"

def decode_to_ndarray(image) -> np.ndarray:
    if os.path.exists(image):
        return np.array(Image.open(image))
    elif type(image) is str:
        return np.array(decode_base64_to_image(image))
    elif type(image) is Image.Image:
        return np.array(image)
    elif type(image) is np.ndarray:
        return np.ndarray
    else:
        Exception("Not an image")
def decode_to_pil(image):
    if os.path.exists(image):
        return Image.open(image)
    elif type(image) is str:
        return decode_base64_to_image(image)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        Exception("Not an image")
def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image).decode()
    elif type(image) is np.ndarray:
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    else:
        Exception("Invalid type")
# ==========================================================
# Modified from mmcv
# ==========================================================


class BaseFileHandler(metaclass=ABCMeta):
    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


class JsonHandler(BaseFileHandler):
    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)


class PickleHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super(PickleHandler, self).dump_to_path(obj, filepath, mode="wb", **kwargs)


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(inpaint_anything_api)
except:
    print("SAM Web UI API failed to initialize")
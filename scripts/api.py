from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Optional, List
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

def inpaint_anything_api(_: gr.Blocks, app: FastAPI):
    class RespResult(BaseModel):
        code: int = 0
        msg: str = ''
        @classmethod
        def failed(cls, msg: str = 'failed'):
            return RespResult(code=-1, msg=msg)
        @classmethod
        def success(cls):
            return RespResult(code=0, msg='Success')
    @app.get("/inpaint-anything/heartbeat")
    async def heartbeat():
        return RespResult.success()
    class SamPredictRequest(BaseModel):
        input_image: str
        sam_model_name: str = "sam_vit_h_4b8939.pth"
        anime_style_chk: bool=False
    class SamPredictResp(RespResult):
        output: str = ''
        @classmethod
        def success(cls, output):
            return SamPredictResp(code=0, output=output)
    @app.post("/inpaint-anything/sam/image")
    async def run_sam(payload: SamPredictRequest = Body(...)) -> Any:
        print(f"inpaint-anything API /inpaint-anything/sam/image received request")

        global sam_dict
        sam_model_id = payload.sam_model_name
        input_image = payload.input_image
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
        del sam_masks
        return SamPredictResp.success(encode_to_base64(seg_img))

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

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(inpaint_anything_api)
except:
    print("SAM Web UI API failed to initialize")
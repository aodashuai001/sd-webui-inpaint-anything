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
import base64
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
from io import BytesIO
import os
import gc

import json
import pickle
import requests
from datetime import datetime, timedelta

from abc import ABCMeta, abstractmethod


# ===========================
# Rigister handler
# ===========================



from ia_file_manager import IAFileManager, download_model_from_hf, ia_file_manager
from ia_config import IAConfig, get_ia_config_index, set_ia_config, setup_ia_config_ini
from ia_get_dataset_colormap import create_pascal_label_colormap
from ia_sam_manager import get_sam_mask_generator, get_sam_predictor
from ia_threading import (async_post_reload_model_weights, await_backup_reload_ckpt_info,
                          await_pre_reload_model_weights, clear_cache_decorator,
                          offload_reload_decorator)
from ia_ui_items import (get_cleaner_model_ids, get_inp_model_ids, get_padding_mode_names,
                         get_sam_model_ids, get_sampler_names)
from ia_logging import ia_logging

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from modules import shared
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
    class EdgeMaskRequest(BaseModel):
        mask_image_url: str
        edge_size: int = 10
    @app.post("/inpaint-anything/edge/mask")
    async def get_edge_mask(payload: EdgeMaskRequest = Body(...)) -> Any:
        img = download_image(payload.mask_image_url)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*payload.edge_size+1, 2*payload.edge_size+1))

        # 执行膨胀操作以获取外扩10像素的遮罩
        dilated_mask = cv2.dilate(img, kernel, iterations=1)

        # 执行腐蚀操作以获取内缩10像素的遮罩
        eroded_mask = cv2.erode(img, kernel, iterations=1)

        # 从膨胀的遮罩中减去腐蚀的遮罩，得到边缘区域
        edge_mask = cv2.subtract(dilated_mask, eroded_mask)
        return RespResult.success(encode_to_base64(edge_mask))
    class SamPredictRequest(BaseModel):
        image_id: int
        input_image: str
        sam_model_name: str = "sam_vit_h_4b8939.pth"
        anime_style_chk: bool=False

    class SamPredictResp(BaseModel):
        segimg: str = ''
        # saminfo: Optional[Any] = None
    @app.post("/inpaint-anything/sam/embedding")
    async def run_sam_embedding(payload: SamPredictRequest = Body(...)) -> Any:
        print(f"inpaint-anything API /inpaint-anything/sam/embedding received request")
        sam_model_id = payload.sam_model_name
        # sam_model_id = 'sam_vit_b_01ec64.pth'
        input_image = decode_to_cv2(payload.input_image)
        sam_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
        if not os.path.isfile(sam_checkpoint):
            return RespResult.failed(f"{sam_model_id} not found, please download")
        if input_image is None:
            return RespResult.failed("Input image not found")

        set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, sam_model_id, IAConfig.SECTIONS.USER)

        ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")
        sam_predictor = get_sam_predictor(sam_checkpoint)
        ia_logging.info(f"{sam_predictor.__class__.__name__} {sam_model_id}")
        sam_predictor.set_image(input_image)
        try:
            image_embedding = sam_predictor.get_image_embedding().cpu().numpy()
        except Exception as e:
            print(traceback.format_exc())
            ia_logging.error(str(e))
            del sam_predictor
            return RespResult.failed("SAM predictor failed")
        # new_length = int(np.ceil(len(image_embedding) / 4) * 4)
        # image_embedding = np.resize(image_embedding, new_length)
        image_embedding_dumps = image_embedding.tobytes()
        # 将二进制字符串编码为Base64格式
        image_embedding_base64 = base64.b64encode(image_embedding_dumps).decode('utf-8')
        # print(image_embedding_base64)
        return RespResult.success(data=[image_embedding_base64])

    def base64_encode(data):
        """
        解决base64编码结尾缺少=报错的问题
        """
        missing_padding = 4 - len(data) % 4
        if missing_padding:
            encode += '=' * missing_padding
        decode = base64.b64encode(encode)
        return decode
    @app.post("/inpaint-anything/sam/mask")
    async def run_sam_mask(payload: SamPredictRequest = Body(...)) -> Any:
        print(f"inpaint-anything API /inpaint-anything/sam/all received request")

        global sam_dict
        sam_model_id = payload.sam_model_name
        input_image = decode_to_ndarray(payload.input_image)
        input_image = remove_alpha_channel(input_image)
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

        ia_logging.info("sam_masks: {}, {}".format(len(sam_masks),np.array(sam_masks[0]["segmentation"]).shape))
        # sam_masks = sorted(sam_masks, key=lambda x: np.sum(x.get("segmentation").astype(np.uint32)))

        # sam_dict["sam_masks"] = copy.deepcopy(sam_masks)
        # print(sam_masks)
        for idx, seg_dict in enumerate(sam_masks):

            # 将字节对象转换为Base64字符串  
            base64_str = bool_array_to_base64(np.array(seg_dict["segmentation"]))

            # 使用pickle序列化数组
            # serialized_array = pickle.dumps(seg_dict["segmentation"])

            # 将序列化后的字节流编码为Base64
            # encoded_array = base64.b64encode(serialized_array)

            # 将Base64字节字符串解码为普通字符串（如果需要）
            # encoded_str = encoded_array.decode('utf-8')
            seg_dict["segmentation"] = base64_str
        # print(sam_masks)
        # del sam_masks
        return RespResult.success(data=sam_masks)
    def bool_array_to_base64(bool_array):
        new_arr = np.argwhere(bool_array)
        bytes_io = BytesIO()
        # 使用numpy.save将数组保存到BytesIO对象中
        np.save(bytes_io, new_arr)
        # 获取保存的数据作为字节
        bytes_data = bytes_io.getvalue()
        # 对字节数据进行Base64编码
        base64_encoded = base64.b64encode(bytes_data)
        # 将Base64编码的字节解码为字符串，以便于显示或存储
        base64_string = base64_encoded.decode('utf-8')
        return base64_string
    @app.post("/inpaint-anything/sam/all")
    async def run_sam_all(payload: SamPredictRequest = Body(...)) -> Any:
        print(f"inpaint-anything API /inpaint-anything/sam/all received request")

        global sam_dict
        sam_model_id = payload.sam_model_name
        input_image = decode_to_ndarray(payload.input_image)
        input_image = remove_alpha_channel(input_image)
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
        # sam_dict["sam_masks"] = copy.deepcopy(sam_masks)
        # print(sam_masks)
        for idx, seg_dict in enumerate(sam_masks):
            seg_img = seg_dict["segmentation"].astype(np.uint8) * 255
            seg_dict["segmentation"] = encode_to_base64(seg_img)
        # print(sam_masks)
        # del sam_masks
        return RespResult.success(data=sam_masks)
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
        # sam_dict["sam_masks"] = copy.deepcopy(sam_masks)
        save_segmentations(sam_masks, payload.image_id)
        # del sam_masks
        return RespResult.success(data=SamPredictResp(segimg=encode_to_base64(seg_img)))
    def save_segmentations(sam_masks, image_id):
        save_name = "_".join([str(image_id), "segmentations"]) + ".dat"
        save_name = os.path.join(outputs_dir(), save_name)
        print(save_name)
        ph = PickleHandler()
        ph.dump_to_path(sam_masks, save_name)
    def load_segmentations(image_id) -> Any:
        for idx in range(5):
            date = datetime.now() - timedelta(days=idx)
            save_name = "_".join([str(image_id), "segmentations"]) + ".dat"
            save_name = os.path.join(outputs_dir(date), save_name)
            if os.path.isfile(save_name):
                ph = PickleHandler()
                return ph.load_from_path(save_name)

    class SamSelectMaskRequest(BaseModel):
        image_id: int
        input_image: str
        select_points: list
        invert_chk: Optional[bool]=False
        ignore_black_chk: Optional[bool]=False
        expand_mask: Optional[int] = 0
    class SamMaskResp(BaseModel):
        mask: str = ''
        image: str = ''
    @app.post("/inpaint-anything/sam/mask")
    async def select_mask(payload: SamSelectMaskRequest = Body(...)) -> Any:
        ignore_black_chk = payload.ignore_black_chk
        # global sam_dict
        sam_masks = load_segmentations(payload.image_id)
        if sam_masks is None:
            ia_logging.info("SAM select task failed, sam_masks is None")
            return RespResult.failed("SAM select task failed")
        # sam_masks = sam_dict["sam_masks"]

        input_image = decode_to_ndarray(payload.input_image)
        image = decode_to_ndarray(payload.input_image)
        mask = np.zeros(image.shape[:2] + (1,), dtype=np.uint8)
        # selected_mask = np.zeros((*image.shape[:2], 1), dtype=bool)
        selected_points = np.array(payload.select_points)

        # 将选定点的掩码设置为 True
        # selected_mask[selected_points[:, 1], selected_points[:, 0]] = True

        # 使用掩码将选定的元素替换为255
        mask[selected_points[:, 1], selected_points[:, 0]] = 255
        
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

        if payload.invert_chk:
            seg_image = np.logical_not(seg_image.astype(bool)).astype(np.uint8) * 255

        # sam_dict["mask_image"] = seg_image

        # if image is not None and image.shape == seg_image.shape:
        #     ret_image = cv2.addWeighted(image, 0.5, seg_image, 0.5, 0)
        # else:
        #     ret_image = seg_image
        
        # expand_mask
        if payload.expand_mask > 0:
            seg_image = expand_mask(seg_image, payload.expand_mask)
            # if input_image is not None and input_image.shape == new_sel_mask.shape:
            #     ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
            # else:
            #     ret_image = new_sel_mask
        # print(seg_image)
        # run_get_mask
        mask_image_base64 = encode_to_base64(seg_image)

        # run_get_alpha_image
        
        alpha_image = Image.fromarray(input_image).convert("RGBA")
        mask_image = Image.fromarray(seg_image).convert("L")

        alpha_image.putalpha(mask_image)


        return RespResult.success(data=SamMaskResp(mask=mask_image_base64, image=encode_to_base64(alpha_image)))

    def expand_mask(new_sel_mask, expand_iteration=1):
        # expand_mask
        # if sam_dict["mask_image"] is None:
        #     return None

        # new_sel_mask = sam_dict["mask_image"]

        expand_iteration = int(np.clip(expand_iteration, 1, 5))

        new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

        sam_dict["mask_image"] = new_sel_mask

        return new_sel_mask

def remove_alpha_channel(image_array):
    """
    检测图片是否有alpha通道，如果有则转换为不带alpha通道的RGB图片。
    
    :param image_array: 输入图片的ndarray。
    :return: 不带alpha通道的RGB图片的ndarray。
    """
    # 将ndarray转换为Pillow Image对象
    image = Image.fromarray(image_array)
    
    # 检查图片是否有alpha通道
    if image.mode == 'RGBA':
        # 删除alpha通道
        rgb_image = image.convert('RGB')
        # 将Pillow Image对象转换回ndarray
        rgb_array = np.array(rgb_image)
        return rgb_array
    else:
        # 如果没有alpha通道，直接返回原ndarray
        return image_array
def download_image(url):
    # 发送HTTP GET请求下载图片
    response = requests.get(url)
    # 确保请求成功
    if response.status_code == 200:
        # 将从url获取的内容转换为numpy数组
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        # 使用OpenCV转换数组为图片对象
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        # 如果请求失败，抛出异常
        response.raise_for_status()
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


def decode_to_cv2(image):
    if os.path.exists(image):
        return cv2.imread(image)
    elif type(image) is str:
        while len(image) % 4 != 0:
            image += '='
        img_data = base64.b64decode(image)
        img_array = np.fromstring(img_data, np.uint8)
        return cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    elif type(image) is Image.Image:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif type(image) is np.ndarray:
        return cv2.imdecode(image, cv2.COLOR_RGB2BGR)
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
def outputs_dir(d = None) -> str:
    """Get inpaint-anything outputs directory.

    Returns:
        str: inpaint-anything outputs directory
    """
    if not d :
        d = datetime.now()
    _ia_outputs_dir = update_ia_outputs_dir(d)
    if not os.path.isdir(_ia_outputs_dir):
        os.makedirs(_ia_outputs_dir, exist_ok=True)
    return _ia_outputs_dir
def update_ia_outputs_dir(d = datetime.now()) -> None:
    """Update inpaint-anything outputs directory.

    Returns:
        None
    """
    config_save_folder = shared.opts.data.get("inpaint_anything_save_folder", "inpaint-anything")
    if config_save_folder in ["inpaint-anything", "img2img-images"]:
        return os.path.join(shared.data_path,
                                            "outputs", config_save_folder,
                                            d.strftime("%Y-%m-%d"))

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
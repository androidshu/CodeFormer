import collections
import math
import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque

from basicsr.utils.registry import ARCH_REGISTRY
import inference_codeformer
from inference_codeformer import *
from RealESRGAN import RealESRGAN


class VideoIterator:

    def __init__(self, reader):
        self.video_reader = reader

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.video_reader.get_frame()
        except Exception as ex:
            print(f'get frame error:{ex}')
        return None


def upsampler(device, args, result_root, input_img_it, total_img_count, base_offest=0, img_base_name=None, restore_img_dqueue=None):
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    # ------------------ set up background upsampler ------------------
    # if args.bg_upsampler == 'realesrgan':
    #     bg_upsampler = set_realesrgan(device, args)
    # else:
    #     bg_upsampler = None

    # -------------------- start to processing ---------------------
    i = base_offest
    for img in tqdm(input_img_it, desc="img list", total=total_img_count):
        if i - base_offest >= total_img_count:
            break
        if img is None:
            continue
        i += 1
        sr_image = model.predict(img)
        basename = str(i).zfill(6)

        restore_img_dqueue.append(RestoreImageWrapper(restored_img=sr_image))
        if args.debug or restore_img_dqueue is None:
            if args.suffix is not None:
                basename = f'{basename}_{args.suffix}'
            save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
            imwrite(sr_image, save_restore_path)

    restore_img_dqueue.append(RestoreImageWrapper(is_ended=True))


if __name__ == '__main__':
    start_time = time.time()
    args = inference_codeformer.parse_argument()

    # ------------------------ input & output ------------------------
    video_iterator = None
    video_reader = None
    fps = None
    audio = None
    result_root = None
    video_name = None
    if args.input_path.lower().endswith(('.mp4', '.mov', '.avi', '.m3u8', '.m3u')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter

        video_reader = VideoReader(args.input_path)
        video_iterator = VideoIterator(video_reader)
        minWH = min(video_reader.width, video_reader.height)
        if minWH <= 640:
            args.upscale = 4
        elif minWH <= 1280:
            args.upscale = 2
        else:
            args.upscale = 1

        audio = video_reader.get_audio()
        fps = video_reader.get_fps() if args.save_video_fps is None else args.save_video_fps
        video_name = os.path.basename(args.input_path).split(".")[0]
        result_root = f'results/{video_name}_{args.upscale}'
    else: # input img folder
        print("please input mp4 file")
        exit(1)

    if args.output_path is not None: # set output path
        result_root = args.output_path
    restore_img_dqueue = collections.deque()
    device = get_device()
    # torch.backends.cudnn.benchmark = True
    thread_pool = ThreadPoolExecutor()
    video_save_feature = thread_pool.submit(inference_codeformer.save_as_video_async, args, result_root, video_name, fps, audio, restore_img_dqueue)

    upsampler(device, args, result_root, iter(video_iterator), total_img_count=video_reader.nb_frames, img_base_name=video_name, restore_img_dqueue=restore_img_dqueue)
    video_save_feature.result()
    if video_reader is not None:
        video_reader.close()
    thread_pool.shutdown()
    end_time = time.time()
    print('\nAll results are saved in {}, all cost time:{:.2f}秒'.format(result_root, end_time - start_time))

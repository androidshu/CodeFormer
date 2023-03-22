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


if __name__ == '__main__':

    args = inference_codeformer.parse_argument()

    # ------------------------ input & output ------------------------
    video_iterator = None
    video_reader = None
    fps = None
    audio = None
    if args.input_path.lower().endswith(('.mp4', '.mov', '.avi', '.m3u8', '.m3u')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter

        video_reader = VideoReader(args.input_path)
        video_iterator = VideoIterator(video_reader)
        minWH = min(video_reader.width, video_reader.height)
        if minWH <= 640:
            args.face_upsample = True
            args.bg_upsampler = "realesrgan"
            args.upscale = 2
            args.fidelity_weight = 1.0
        else:
            args.upscale = 1

        audio = video_reader.get_audio()
        fps = video_reader.get_fps() if args.save_video_fps is None else args.save_video_fps
        video_name = os.path.basename(args.input_path).split(".")[0]
        result_root = f'results/{video_name}_{args.fidelity_weight}'
    else: # input img folder
        print("please input mp4 file")
        exit(1)

    if not args.output_path is None: # set output path
        result_root = args.output_path
    # test_img_num = len(input_img_list)
    # if test_img_num == 0:
    #     raise FileNotFoundError('No input image/video is found...\n'
    #         '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=inference_codeformer.pretrain_model_url['restoration'],
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    start_time = time.time()
    restore_img_dqueue = collections.deque()
    print("start")
    # # concurrent
    # img_count = len(input_img_list)
    # thread_pool = ThreadPoolExecutor()
    # cuda_count = torch.cuda.device_count()
    # if cuda_count > 1:
    #     step = math.ceil(img_count / args.thread_count)
    #     feature_list = []
    #     for i in range(cuda_count):
    #         device = get_device(i)
    #         start = i * step
    #         end = min((i + 1) * step, img_count)
    #         feature = thread_pool.submit(restore_face_and_upsampler, device, checkpoint, input_img_list[start: end], start, restore_img_dqueue=restore_img_dqueue)
    #         feature_list.append(feature)
    #
    #     for feature in feature_list:
    #         feature.result()
    #
    #     thread_pool.shutdown()
    # else:
    device = get_device()
    # torch.backends.cudnn.benchmark = True
    thread_pool = ThreadPoolExecutor()
    video_save_feature = thread_pool.submit(inference_codeformer.save_as_video_async, args, result_root, video_name, fps, audio, restore_img_dqueue)

    inference_codeformer.restore_face_and_upsampler(device, checkpoint, args, result_root, iter(video_iterator), total_img_count=video_reader.nb_frames, img_base_name=video_name, restore_img_dqueue=restore_img_dqueue)
    video_save_feature.result()
    if video_reader is not None:
        video_reader.close()
    thread_pool.shutdown()
    end_time = time.time()
    print('\nAll results are saved in {}, all cost time:{:.2f}秒'.format(result_root, end_time - start_time))

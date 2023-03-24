import math
import os
import sys
import hashlib
import ffmpeg

from tqdm import tqdm
from moviepy.editor import VideoFileClip

import video_inference_codeformer
import shutil


def convert_url_to_path(url, file_dir='temp'):
    md5 = hashlib.md5(url.encode()).hexdigest()
    basename = os.path.basename(url).split("?")[0]
    name = md5 + "_" + basename

    current_dir = os.path.dirname(sys.argv[0])
    dir_path = os.path.join(current_dir, file_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.join(dir_path, name)


def download_to_path(url, file_path):
    command = f'ffmpeg -i "{url}" -codec copy "{file_path}"'
    print(f"down video file, command:{command}")
    os.system(command)


if __name__ == '__main__':

    input_urls_file_path = sys.argv[1]

    with open(input_urls_file_path, "r", encoding="UTF-8") as f:
        url_list = f.readlines()

    if len(sys.argv) >= 4:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        url_list = url_list[start: min(end, len(url_list))]
        
    input_urls_file_name = os.path.basename(input_urls_file_path).split('.')[0]
    out_mp4_collect_dir = f'results/{input_urls_file_name}'
    if not os.path.isdir(out_mp4_collect_dir):
        os.makedirs(out_mp4_collect_dir)

    for url in tqdm(url_list):
        url = url.replace("\n", '').replace("\r", '')
        path = url

        url_list = url_list
        if url.startswith("http://") or url.startswith("https://"):
            path = convert_url_to_path(url)
            if not os.path.exists(path):
                download_to_path(url, path)
        if not os.path.exists(path):
            print(f'handle video failed, file path:{path} not exists.')
            continue
        # check the out mp4
        file_only_name = os.path.basename(path).split('.')[0]
        out_mp4_file_name = f'{file_only_name}_1.0/{file_only_name}.mp4'
        out_mp4_path = os.path.join(".", f'results/{out_mp4_file_name}')
        if os.path.exists(out_mp4_path):
            try:
                input_mp4_duration = int(VideoFileClip(path).duration)
                out_mp4_duration = int(VideoFileClip(out_mp4_path).duration)
                if out_mp4_duration > 0 and input_mp4_duration > 0 and abs(
                        out_mp4_duration - input_mp4_duration) < 100:
                    print('output mp4 already exists, and time correct')

                    collect_file_path = os.path.join(out_mp4_collect_dir, f'{file_only_name}.mp4')
                    if not os.path.exists(collect_file_path):
                        success = shutil.copyfile(out_mp4_path, collect_file_path)
                        print(f'copy file to collect dir success:{success}')
                    continue
            except Exception as e:
                print("parse mp4 error")

        os.system(f"python video_inference_codeformer.py --input_path {path}")



import math
import os
import sys
import hashlib
import ffmpeg

from tqdm import tqdm

import inference_video_codeformer


def convert_url_to_path(url, file_dir='temp'):
    md5 = hashlib.md5(url.encode()).hexdigest()
    basename = os.path.basename(url)
    name = md5 + "_" + basename

    current_dir = os.path.dirname(sys.argv[0])
    dir_path = os.path.join(current_dir, file_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.join(dir_path, name)


def download_to_path(url, file_path):
    command = f"ffmpeg -i {url} {file_path}"
    print(f"down video file, command:{command}")
    os.system(command)


if __name__ == '__main__':

    filePath = sys.argv[1]
    with open(filePath, "r", encoding="UTF-8") as f:
        url_list = f.readlines()

    for url in tqdm(url_list):
        url = url.replace("\n", '').replace("\r", '')
        path = url

        if url.startswith("http://") or url.startswith("https://"):
            path = convert_url_to_path(url)
            if not os.path.exists(path):
                download_to_path(url, path)
        if not os.path.exists(path):
            print(f'handle video failed, file path:{path} not exists.')
            continue
        os.system(f"python inference_video_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path {path}")



# -*- coding: utf-8 -*-
import jieba
import multiprocessing
import os
import re
import requests
from dashscope import ImageSynthesis
from datetime import datetime
from http import HTTPStatus
from moviepy.config import change_settings
from moviepy.editor import *
from moviepy.video.fx.all import fadein, fadeout
from pathlib import Path
from typing import List
import time

import utils

# 初始化配置
ALI_API_KEY = "sk-8a67588417f64ad188403ba3cede211a"
FILE_PATH = "resources/text_file.txt"
API_TOKEN = "sk-mxwtcsithzksgmgxumqakjubzgcksuriyxcxfvpjwfwzxvff"
# 添加系统字体路径到 MoviePy 配置
font_paths = [
    "/usr/share/fonts",          # Linux
    "/Library/Fonts",            # macOS 系统字体
    os.path.expanduser("~/Library/Fonts"),  # macOS 用户字体
    "C:/Windows/Fonts"           # Windows
]
change_settings({"FONTPATH": ":".join(font_paths)})
# Apple Silicon (M1/M2 芯片)
change_settings({"IMAGEMAGICK_BINARY": "/opt/homebrew/bin/magick"})
# Windows NVIDIA
# change_settings({"IMAGEMAGICK_BINARY": "C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

CONFIG = {
    "output_dir": "output",
    "temp_dir": "temp",
    "video_res": (576, 1024),
    "font": "Source Han Serif SC",
    "font_size": 40,
    "text_color": "white",
    "bg_color": "rgba(0,0,0,0.5)",
    "fps": 24,
    "transition": {
        "type": "fade",
        "duration": 0.5
    }
}


import sys
import platform

def set_imagemagick_config():
    config = {}

    # macOS 系统判断
    if sys.platform == 'darwin':
        # Apple Silicon (M1/M2) 架构检测
        if platform.machine() == 'arm64':
            config["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"
        # Intel 芯片 macOS
        else:
            config["IMAGEMAGICK_BINARY"] = "/usr/local/bin/magick"

    # Windows 系统判断
    elif sys.platform.startswith('win32'):
        config["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

    # 应用配置
    if config:
        change_settings(config)

def setup_directories():
    """创建输出目录"""
    os.makedirs(CONFIG['temp_dir'], exist_ok=True)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

def split_text(text: str) -> list:
    """智能分句处理

    Args:
        text: 输入的文本字符串
    Returns:
        list: 分句后的句子列表
    """
    # 使用正则表达式进行分句
    pattern = r'([。！？；\.\!?;]\s*)'
    sentences = re.split(pattern, text)
    sentences = [''.join(group) for group in zip(sentences[0::2], sentences[1::2])]

    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:  # 跳过空字符串
            continue
        # 使用jieba进行分词，只是为了检查句子长度
        words = list(jieba.cut(sentence, cut_all=False))
        if len(words) < 3:  # 对于过短的句子，跳过
            continue
        result.append(sentence)

    return result

def split_audio_text(text: str) -> list:
    """智能分句处理

    Args:
        text: 输入的文本字符串
    Returns:
        list: 分句后的句子列表，不包含标点符号
    """
    # 使用正则表达式进行分句，包括逗号
    pattern = r'([，。！？；,\.\!?;]\s*)'
    sentences = re.split(pattern, text)

    # 合并分割后的句子和标点
    sentences = [''.join(group) for group in zip(sentences[0::2], sentences[1::2])]

    result = []
    for sentence in sentences:
        # 去除所有标点符号
        clean_sentence = re.sub(r'[^\w\s]', '', sentence).strip()
        if not clean_sentence:  # 跳过空字符串
            continue
        result.append(clean_sentence)

    return result

def generate_speech(text: str, filename: str) -> float:
    """生成语音并返回时长"""
    url = "https://api.siliconflow.cn/v1/audio/speech"
    payload = {
        "model": "RVC-Boss/GPT-SoVITS",
        "input": text,
        "voice": "RVC-Boss/GPT-SoVITS:david",
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": True,
        "speed": 3,
        "gain": 0
    }
    headers = {
        "Authorization": f"Bearer {API_TOKEN}", # 从 https://cloud.siliconflow.cn/account/ak 获取
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # 打印响应状态码和响应内容
    print(response.status_code)
    # 获取音频时长
    audio = AudioFileClip(filename)
    duration = audio.duration
    print(f"duration: {duration}")
    audio.close()
    return duration

def generate_comic_image_by_API(prompt: str, filename: str) -> str:
    print('----sync call, please wait a moment----')
    response = ImageSynthesis.call(api_key=ALI_API_KEY,
                              model="wanx2.1-t2i-turbo",
                              prompt=prompt,
                              n=1,
                              seed = 4294967290,
                              size='576*1024')
    print('response: %s' % response)
    if response.status_code == HTTPStatus.OK:
        # 在当前目录下保存图片
        for result in response.output.results:
            with open(filename, 'wb+') as f:
                f.write(requests.get(result.url).content)
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))
    return filename

def generate_comic_image_from_local_model(prompt: str, filename: str) -> str:
    print('----Generating pictures using local machine----')
    utils.invoke_txt2img_model(prompt, filename)
    return filename

def create_subtitle_clip(text: str, duration: float) -> TextClip:
    """创建带背景框的字幕"""
    return TextClip(
        text,
        font=CONFIG['font'],
        fontsize=CONFIG['font_size'],
        color=CONFIG['text_color'],
        bg_color=CONFIG['bg_color'],
        size=(CONFIG['video_res'][0]*0.9, None),
        method='caption'
    ).set_duration(duration).set_position(('center', 'bottom'))

def generate_video(text: str, output_file: str = "output.mp4"):
    """主生成函数"""
    setup_directories()

    # 分句处理 for image
    sentences = split_text(text)
    clips = []
    duration = 0
    video_clip = None

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx+1}/{len(sentences)}: {sentence}")

        # 生成漫画图片
        image_file = os.path.join(CONFIG['temp_dir'], f"image_{idx}.png")
        generate_comic_image_from_local_model(sentence, image_file)

        small_audio_sentence = split_audio_text(sentence)
        for small_idx, small_sentence in enumerate(small_audio_sentence):
            print(f"Processing small sentence {small_idx+1}/{len(small_audio_sentence)}: {small_sentence}")
            # 生成语音
            audio_file = os.path.join(CONFIG['temp_dir'], f"audio_{idx}_{small_idx}.mp3")
            duration = generate_speech(small_sentence, audio_file)

            # 创建视频片段
            img_clip = ImageClip(image_file).set_duration(duration)
            audio_clip = AudioFileClip(audio_file)
            subtitle_clip = create_subtitle_clip(small_sentence, duration)

            # 合成带音频和字幕的片段
            video_clip = CompositeVideoClip([img_clip, subtitle_clip])
            video_clip = video_clip.set_audio(audio_clip)

            # 添加转场特效
            if idx > 0 and small_idx <= 0:
                video_clip = video_clip.fx(fadein, CONFIG['transition']['duration'])
                clips[-1] = clips[-1].fx(fadeout, CONFIG['transition']['duration'])
            clips.append(video_clip)

    # 合成最终视频
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        os.path.join(CONFIG['output_dir'], output_file),
        fps=CONFIG['fps'],
        codec="libx264",
        audio_codec="aac"
    )


    # 清理临时文件
    [os.remove(os.path.join(CONFIG['temp_dir'], f))
     for f in os.listdir(CONFIG['temp_dir'])]

def process_single_file(file_path: str, num: str):
    """示例处理函数（根据实际需求修改）"""
    with open(file_path, "r", encoding="utf-8") as f:
        sample_text = f.read()
    generate_video(sample_text, f"comic_video_{num}.mp4")
    return len(sample_text)

def split_large_file(file_path: str, output_dir: str, max_words=2000) -> List[str]:
    """分割大文件为多个小文件[7]"""
    chunks = []
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        words = f.read().split()

    for i in range(0, len(words), max_words):
        chunk = words[i:i+max_words]
        output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_part{i//max_words}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(" ".join(chunk))
        chunks.append(output_path)
    return chunks

def parallel_job_processor(input_dir: str, output_dir: str, workers=4):
    """并行任务处理框架[2]"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 分割文件
    all_chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            all_chunks.extend(split_large_file(file_path, output_dir))

    # 2. 并行处理
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_single_file, all_chunks)

    return results

if __name__ == "__main__":
    start_time = time.time()
    set_imagemagick_config()
    process_single_file(FILE_PATH, f"{datetime.now().strftime('%y%m%d-%H-%M-%S')}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"耗时: {elapsed_time:.6f} 秒")

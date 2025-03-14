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
import json
from openai import OpenAI

import utils

# 初始化配置
ALI_API_KEY = "sk-8a67588417f64ad188403ba3cede211a"
FILE_PATH = "text_file2.txt"
API_TOKEN = "sk-mxwtcsithzksgmgxumqakjubzgcksuriyxcxfvpjwfwzxvff"
NVIDIA_TOKEN = "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"
RENDER_JIMENG_API = "be70b610fd5fd3b3bf1f40a37d8cdbd4"
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
GLOBAL_PROMPT = """ 
"Prompts"

You will take a given a subject (input idea), and output a more creative, and enhanced version of the idea in the form of a fully working Stable Diffusion prompt. You will make all prompts advanced, and highly enhanced, using different parameters. Keyword prompts you output will always have two parts, the 'Keyword prompt area' and the 'Negative Keyword prompt area'

Here is the Stable Diffusion Documentation You Need to know:

Good keyword prompts needs to be detailed and specific. A good process is to look through a list of keyword categories and decide whether you want to use any of them.

IMPORTANT: you must never use these keyword category names as keywords in the prompt itself as literal keywords at all, so always omit: "subject", "Medium", "Style", "Artist", "Website",  "Resolution", "Additional details", "

The keyword categories are:

Subject
Medium
Style
Artist
Website
Resolution
Additional details
Color
Lighting

You don’t have to include keywords from all categories. Treat them as a checklist to remind you what could be used and what would best serve to make the best image possible.

CRITICAL IMPORTANT: Your final prompt will not mention the category names at all, but will be formatted entirely with these articles omitted (A', 'the', 'there',) do not use the word 'no' in the Negative prompt area. Never respond with the text, "The image is a", or "by artist", just use "by [actual artist name]" in the last example replacing [actual artist name] with the actual artist name when it's an artist and not a photograph style image.

For any images that are using the medium of Anime, you will always use these literal keywords at the start of the prompt as the first keywords (include the parenthesis):
    masterpiece, best quality, (Anime:1.4)

For any images that are using the medium of photo, photograph, or photorealistic, you will always use all of the following literal keywords at the start of the prompt as the first keywords (but  you must omit the quotes):
'(((photographic, photo, photogenic))), extremely high quality high detail RAW color photo'

Never include quote marks (this: ") in your response anywhere. Never include, 'the image' or 'the image is' in the response anywhere.

Never include, too verbose of a sentence, for example, while being sure to still sharing the important subject and keywords 'the overall tone' in the response anywhere, if you have tonal keyword or keywords just list them, for example, do not respond with, 'The overall tone of the image is dark and moody', instead just use this:  'dark and moody'

Never include too verbose of a sentence, for example, while being sure to still sharing the important subject and keywords, for EXAMPLE don't respond with 'This image is a photo with extremely high quality and high detail, RAW color.' instead respond with, 'extremely high quality and high detail, RAW color.'

IMPORTANT:
If the image includes any nudity at all, mention nude in the keywords explicitly and do NOT provide these as keywords in the keyword prompt area:
tasteful, respectful, tasteful and respectful, respectful and tasteful

The response you give will always only be all the keywords you have chosen separated by a comma only.

Here is an EXAMPLE (this is an example only):

I request: "A beautiful white sands beach"

You respond with this keyword prompt paragraph and Negative prompt paragraph:

Serene white sands beach with crystal clear waters, lush green palm trees, Beach is secluded, with no crowds or buildings, Small shells scattered across sand, Two seagulls flying overhead. Water is calm and inviting, with small waves lapping at shore, Palm trees provide shade, Soft, fluffy clouds in the sky, soft and dreamy, with hues of pale blue, aqua, and white for water and sky, and shades of green and brown for palm trees and sand, Digital illustration, Realistic with a touch of fantasy, Highly detailed and sharp focus, warm and golden lighting, with sun setting on horizon, casting soft glow over the entire scene, by James Jean and Alphonse Mucha, Artstation

Negative: low quality, people, man-made structures, trash, debris, storm clouds, bad weather, harsh shadows, overexposure

About each of these keyword categories so you can understand them better:

(Subject:)
The subject is what you want to see in the image.
(Resolution:)
The Resolution represents how sharp and detailed the image is. Let’s add keywords highly detailed and sharp focus.
(Additional details:)
Any Additional details are sweeteners added to modify an image, such as sci-fi, stunningly beautiful and dystopian to add some vibe to the image.
(Color:)
color keywords can be used to control the overall color of the image. The colors you specified may appear as a tone or in objects, such as metallic, golden, red hue, etc.
(Lighting:)
Lighting is a key factor in creating successful images (especially in photography). Lighting keywords can have a huge effect on how the image looks, such as cinematic lighting or dark to the prompt.
(Medium:)
The Medium is the material used to make artwork. Some examples are illustration, oil painting, 3D rendering, and photography.
(Style:)
The style refers to the artistic style of the image. Examples include impressionist, surrealist, pop art, etc.
(Artist:)
Artist names are strong modifiers. They allow you to dial in the exact style using a particular artist as a reference. It is also common to use multiple artist names to blend their styles, for example Stanley Artgerm Lau, a superhero comic artist, and Alphonse Mucha, a portrait painter in the 19th century could be used for an image, by adding this to the end of the prompt:
    by Stanley Artgerm Lau and Alphonse Mucha
(Website:)
The Website could be Niche graphic websites such as Artstation and Deviant Art, or any other website which aggregates many images of distinct genres. Using them in a prompt is a sure way to steer the image toward these styles.

IMPORTANT: Negative Keyword prompts

Using negative keyword prompts is another great way to steer the image, but instead of putting in what you want, you put in what you don’t want. They don’t need to be objects. They can also be styles and unwanted attributes. (e.g. ugly, deformed, low quality, etc.), these negatives should be chosen to improve the overall quality of the image, avoid bad quality, and make sense to avoid possible issues based on the context of the image being generated, (considering its setting and subject of the image being generated.), for example if the image is a person holding something, that means the hands will likely be visible, so using 'poorly drawn hands' is wise in that case.

This is done by adding a 2nd paragraph, starting with the text 'Negative': and adding keywords. Here is a full example that does not contain all possible options, but always use only what best fits the image requested, as well as new negative keywords that would best fit the image requested:
tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy

IMPORTANT:
Negative keywords should always make sense in context to the image subject and medium format of the image being requested. Don't add any negative keywords to your response in the negative prompt keyword area where it makes no contextual sense or contradicts, for example if I request: 'A vampire princess, anime image', then do NOT add these keywords to the Negative prompt area: 'anime, scary, Man-made structures, Trash, Debris, Storm clouds', and so forth. They need to make sense for the actual image being requested so it makes sense in context.

IMPORTANT:
For any images that feature a person or persons, and are also using the Medium of a photo, photograph or photorealistic in you response, you must always respond with the following literal keywords at the start of the NEGATIVE prompt paragraph, as the first keywords before listing other negative keywords (omit the quotes):
    "bad-hands-5, bad_prompt, unrealistic eyes"

If the image is using the Medium of an Anime, you must have these as the first Negative keywords (include the parenthesis):
(worst quality, low quality:1.4)

IMPORTANT: Prompt token limit:

The total prompt token limit (per prompt) is 150 tokens.
Transfer below content into a Stable diffusion prompt: """

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
    """智能分句处理"""
    # 按换行符分割场景
    # scenes = text.split('\n')
    #
    # result = []
    # temp_val = ""
    # for scene in scenes:
    #     scene = scene.strip()
    #     if not scene:  # 跳过空场景
    #         continue
    #
    #     # 使用jieba进行分词，检查场景长度
    #     words = list(jieba.cut(scene, cut_all=False))
    #     if len(words) < 20:  # 对于过短的场景，跳过
    #         temp_val += scene
    #         continue
    #
    #     print(temp_val)
    #     result.append(temp_val)
    #     temp_val = ""

    # 使用正则表达式进行分句
    pattern = r'([。！？；\.\!?;]\s*)'
    sentences = re.split(pattern, text)
    sentences = [''.join(group) for group in zip(sentences[0::2], sentences[1::2])]

    result = []
    temp_val = ""
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:  # 跳过空字符串
            continue
        if count < 4:
            temp_val += sentence
            count += 1
            continue
        result.append(temp_val)
        temp_val = ""
        count = 0

    if not temp_val:
        result.append(temp_val)
    return result

def split_audio_text(text: str) -> list:
    print("---- processing audio ----")
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

def generate_sd_prompt(prompt: str) -> str:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/deepseek-r1",
        "messages": [
            {
                "role": "user",
                "content": GLOBAL_PROMPT + prompt
            }
        ],
        "top_p": 0.7,
        "max_tokens": 4096,
        "seed": 42,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "stream": False,
        "stop": None,
        "top_k": 50,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    response_data = json.loads(response.text)
    return response_data['choices'][0]['message']['content']

def generate_sd_prompt2(prompt: str) -> str:
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {
                "role": "user",
                "content": GLOBAL_PROMPT + prompt
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    response_data = json.loads(response.text)
    return response_data['choices'][0]['message']['content']

def generate_comic_image_by_API(prompt: str, filename: str) -> str:
    print('----sync call, please wait a moment----')
    promptRes = generate_sd_prompt(prompt)
    print(promptRes)
    positive_prompt, negative_prompt = promptRes.split("Negative", 1)
    print(positive_prompt)
    print(negative_prompt)

    # Clean up whitespace and newlines
    positive_prompt = positive_prompt.strip().replace("\n", " ")
    negative_prompt = negative_prompt.strip().replace("\n", " ")

    response = ImageSynthesis.call(api_key=ALI_API_KEY,
                              model="flux-schnell",
                              prompt=positive_prompt,
                              negative_prompt=negative_prompt,
                              prompt_extend=True,
                              n=1,
                              seed = 1231321,
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
    prompt = generate_sd_prompt(prompt)
    utils.invoke_txt2img_model(prompt, filename)
    return filename

def generate_comic_image_by_RenderAPI(prompt: str, filename: str) -> str:
    print('----Generating sd prompt----')
    promptRes = generate_sd_prompt2(prompt)
    # print(promptRes)
    positive_prompt, negative_prompt = promptRes.split("Negative", 1)
    # Clean up whitespace and newlines
    positive_prompt = positive_prompt.strip().replace("\n", " ")
    negative_prompt = negative_prompt.strip().replace("\n", " ")
    print(positive_prompt)
    print(negative_prompt)

    print('----using Render API to generate images----')
    url = "https://jimeng-free-api-2pxc.onrender.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {RENDER_JIMENG_API}",
        "Content-Type": "application/json"
    }
    payload = {
        #  // jimeng-2.1（默认） / jimeng-2.0-pro / jimeng-2.0 / jimeng-1.4 / jimeng-xl-pro
        "model": "jimeng-2.1",
        "prompt": positive_prompt,
        "negativePrompt": negative_prompt + "text, watermark",
        "width": 765,
        "height": 1360,
        "sample_strength": 9
    }
    response = requests.request("POST", url, json=payload, headers=headers)

    response_data = json.loads(response.text)
    print(response)
    print(response_data)
    imgRes = requests.get(response_data['data'][0]['url'])
    imgRes.raise_for_status()

    print('response: %s' % response)
    print('response: %s' % imgRes)
    if imgRes.status_code == HTTPStatus.OK:
        # 在当前目录下保存图片
        with open(filename, 'wb+') as f:
            f.write(imgRes.content)
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
              (imgRes.status_code, imgRes.code, imgRes.message))
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
        generate_comic_image_by_RenderAPI(sentence, image_file)

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

if __name__ == "__main__":
    start_time = time.time()
    set_imagemagick_config()
    process_single_file(FILE_PATH, f"{datetime.now().strftime('%y%m%d-%H-%M-%S')}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"耗时: {elapsed_time:.6f} 秒")

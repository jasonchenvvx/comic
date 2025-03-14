import requests
import base64
import json
from datetime import datetime
import os

def save_base64_image(base64_str, filepath):
    """将base64字符串保存为图片"""
    try:
        # 解码并保存图片
        image_data = base64.b64decode(base64_str.split(",", 1)[-1])
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filepath

    except Exception as e:
        print(f"保存图片失败: {str(e)}")
        return None

def invoke_txt2img_model(prompt: str, filename: str):
    # prompt_json = json.loads(prompt)
    # # Clean up whitespace and newlines
    # positive_prompt = prompt_json['positive_prompt']
    # negative_prompt = prompt_json['negative_prompt']
    # print(positive_prompt)
    # print(negative_prompt)
    # Split into positive and negative prompts
    print(prompt)
    positive_prompt, negative_prompt = prompt.split("Negative", 1)
    print(positive_prompt)
    print(negative_prompt)

    # Clean up whitespace and newlines
    positive_prompt = positive_prompt.strip().replace("\n", " ")
    negative_prompt = negative_prompt.strip().replace("\n", " ")
    # API配置
    api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    headers = {"Content-Type": "application/json"}

    # 请求参数
    payload = {
        "denoising_strength": 0,
        "prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "seed": 661556997210625,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 35,
        "cfg_scale": 9,
        "width": 576,
        "height": 1024,
        "restore_faces": False,
        "tiling": False,
        "override_settings": {
            "sd_model_checkpoint": "animagineXLV31_v31.safetensors [e3c47aedb0]"
        },
        "script_args": [0, True, True, "LoRA", 1, 1],
        "sampler_name": "DPM++ 2M Karras"
    }

    try:
        # 发送API请求
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # 检查HTTP错误

        # 解析响应
        result = response.json()

        # 检查图像数据
        if "images" not in result or len(result["images"]) == 0:
            raise ValueError("API响应中未找到图像数据")

        # 保存第一张图片
        saved_path = save_base64_image(result["images"][0], filename)

        if saved_path:
            print(f"图片已成功保存至: {saved_path}")
        else:
            print("图片保存失败")

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {str(e)}")
        print(f"响应内容: {e.response.text if e.response else '无响应'}")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    invoke_txt2img_model("hello kitty", "output/20250225.png")
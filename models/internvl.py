# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2024- 08- 07
Authors: Yu wenlong  and  DRAGON_501
Description:
Functions: 
Input: 
Output: 
Note:

Link: 

*************************************Import***********************************"""
import math
import numpy as np
import os
import requests

import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from util.plot_utils import clear_plot, plt_show_save
from crp_lrp.crp.image import imgify
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_image_from_tensor(image, input_size=448, max_num=6):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, torch.Tensor):
        image = (image - image.min()) / (image.max() - image.min())
        image = image * 255
        image_array = image.byte().numpy()
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        image = Image.fromarray(image_array).convert('RGB')
    else:
        raise ValueError('Unsupported image type')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def store_tensor_2_dir(image, input_size=448, max_num=6, store_dir='./tmp_api_imgs/', file_name='image.jpg'):
    if isinstance(image, str):
        pass
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, torch.Tensor):
        image = (image - image.min()) / (image.max() - image.min())
        image = image * 255
        image_array = image.byte().numpy()
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        image = Image.fromarray(image_array).convert('RGB')
    else:
        raise ValueError('Unsupported image type')
    img_dir = os.path.join(store_dir, file_name)
    image.save(img_dir)

    return img_dir


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    model_name = model_name.split('/')[-1]
    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


class internvl2():
    def __init__(self, b_api=False, url=None, api_key=None):
        super().__init__()

        self.b_api = b_api
        if b_api is True:
            if api_key is not None and url is not None:
                self.api_key = api_key
                self.url = url
            else:
                self.api_key = 'Your api key here'
                self.url = 'Your url here'
            self.llm = self.api_llm

        else:
            self.api_key = None
            self.url = None

            path = './model/OpenGVLab/InternVL2-8B'
            # path = './model/OpenGVLab/InternVL2-26B'

            # ***********************************************************************************************
            # *********************************************************************

            b_use_multi_gpu = True

            if b_use_multi_gpu is False:
                self.model = AutoModel.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda:0',
                    low_cpu_mem_usage=True,
                    trust_remote_code=True).eval()

            # *********************************************************************
            # ***********************************************************************************************
            # # Otherwise, you need to set device_map to use multiple GPUs for inference.

            else:
                device_map = split_model(path)
                self.model = AutoModel.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map=device_map).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

            self.generation_config = dict(
                num_beams=1,
                max_new_tokens=1024,
                do_sample=False,
            )
            self.llm = self.model.chat
            response = self.text_infer('Hello, who are you?')
            # print(response)
            # print(f"InternVL2( {path} ) loaded successfully. Using on multi gpus: {b_use_multi_gpu}")

    def inference(self, images=None, prompt=None, history=None, pure_text=False, b_single=False, max_num: int = 6):
        import torch.nn.functional as F

        if isinstance(images, str):
            images = load_image(images, max_num=6).to(torch.bfloat16).cuda()  # [7,3,448, 448]
        if b_single is True:
            pixel_values = images

            if prompt is None:
                 prompt= '<image>\nPlease describe the image. No more than 40 words. ' \
                           'You should describe some attributes of each object in the given image, ' \
                           'such as color, shape, size, etc.' \
                           'Keep your description concise and veritable, rather than imagination or exaggeration.'
            response, history = self.llm(self.tokenizer, pixel_values, prompt, self.generation_config, history=None,
                                           return_history=True)
        else:

            plt_show_save(imgify(images, symmetric=True, grid=(3, 5), padding=False), b_show=False)
            pixel_values_dict = {}

            for i in range(len(images)):
                pixel_values_dict[i] = load_image_from_tensor(images[i], max_num=6).to(torch.bfloat16).cuda()

            pixel_values = torch.cat(list(pixel_values_dict.values()), dim=0)
            num_patches_list = [xxx.size(0) for xxx in pixel_values_dict.values()]
            plt_show_save(imgify(pixel_values.to(torch.float16), symmetric=True, grid=(3, 5), padding=False), b_show=False)

            img_question = ''
            for i in range(len(pixel_values)):
                img_question += f'Image-{i+1}: <image>\n'

            # question += 'Describe the 15 images in detail.'
            if pure_text is False:
                question = img_question + prompt
            else:
                question = prompt

            #########################################################################################
            # *************************************************************************************
            response, history = self.llm(self.tokenizer, pixel_values, question, self.generation_config,
                                           num_patches_list=num_patches_list,
                                           history=history, return_history=True)
            # *************************************************************************************
            #########################################################################################

        torch.cuda.empty_cache()
        return response, history

    def text_infer(self, prompt):
        response, history = self.llm(self.tokenizer, None, prompt, self.generation_config, history=None,
                                       return_history=True)
        # print(f'Assistant: {response}')
        return response

    def api_llm(self, prompt=None):

        question = prompt if prompt is not None else "Describe Tianjin University in China."  # (Question)
        data = {
            'question': question,
            'api_key': self.api_key
        }
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Response:", response.json().get("response", "No response key found in the JSON."))
            else:
                print("Error:", response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        return response


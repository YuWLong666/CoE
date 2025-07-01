# -*- coding:utf-8 -*-

from openai import OpenAI
from pydantic import BaseModel
import base64


"""**********************************Import***********************************"""
'''***************************************************************************'''

GPT_API_KEY = ""
GPT_BASE_URL = ''


class CalendarEvent(BaseModel):
    image_1: list[str]
    image_2: list[str]
    image_3: list[str]
    image_4: list[str]
    image_5: list[str]
    image_6: list[str]
    image_7: list[str]
    image_8: list[str]
    image_9: list[str]
    image_10: list[str]
    image_11: list[str]
    image_12: list[str]
    image_13: list[str]
    image_14: list[str]
    image_15: list[str]


class CalendarEvent_cpeeval(BaseModel):
    score_explanation_1: str
    score_1: int
    score_explanation_2: str
    score_2: int
    score_explanation_3: str
    score_3: int
    total_score: int


class OpenAIGpt():

    def __init__(self, model_name=None, args=None, b_format=True, format_func_name='', api_key=None, base_url=None, temperature=0.1):
        self.api_key = GPT_API_KEY if api_key is None else api_key
        self.base_url = GPT_BASE_URL if base_url is None else base_url
        self.b_format = b_format
        self.format_func_name = format_func_name

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.model_name = "gpt-4o-mini" if model_name is None else model_name
        self.temperature = temperature
        if self.b_format is True:
            if self.format_func_name == 'cpe_eval':
                self.format_func = CalendarEvent_cpeeval
            else:
                self.format_func = CalendarEvent

    def inference(self, images=None, prompt=None, system_prompt=None, messages=None, history=None, pure_text=False, b_single=False,
                  max_num: int = 6, resolution='low', b_show=False, img_store_path=''):
        assert self.api_key is not None, "API Key is not provided"
        assert self.base_url is not None, "Base URL is not provided"
        assert self.model_name is not None, "Model name is not provided"
        assert images is not None and (isinstance(images, dict),
                isinstance(images, list) or isinstance(images, str)), "images should be str of list"

        prompt = 'Please describe the image.' if prompt is None else prompt
        system_prompt = "You are a helpful assistant designed to describe the commonality and " \
                        "specificity of the given images, and output a JSON format response." if system_prompt is None else system_prompt

        if isinstance(images, str):
            image_files = [images]
        elif isinstance(images, dict):
            image_files = [img for img in images.values()]
        else:
            image_files = images

        if messages is None:
            messages = [
                {"role": "system",
                 "content": system_prompt
                 },
                {
                    "role": "user",
                    "content":
                        [{
                        "type": "text",
                        "text": prompt
                    }
                ]}
            ]

            for img_file in image_files:
                base64_image = encode_image(img_file)
                messages[1]['content'].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": resolution
                        }
                    })
        else:
            pass

        # Send the request to OpenAI API
        if self.b_format is True:
            try:
                dict_answer = {}
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=self.format_func,
                )
                answer = response.choices[0].message.parsed

                if self.format_func_name == 'cpe_eval':
                    dict_answer['score_explanation_1'] = answer.score_explanation_1
                    dict_answer['score_1'] = answer.score_1
                    dict_answer['score_explanation_2'] = answer.score_explanation_2
                    dict_answer['score_2'] = answer.score_2
                    dict_answer['score_explanation_3'] = answer.score_explanation_3
                    dict_answer['score_3'] = answer.score_3
                    dict_answer['total_score'] = answer.total_score
                else:
                    dict_answer['image_1'] = answer.image_1
                    dict_answer['image_2'] = answer.image_2
                    dict_answer['image_3'] = answer.image_3
                    dict_answer['image_4'] = answer.image_4
                    dict_answer['image_5'] = answer.image_5
                    dict_answer['image_6'] = answer.image_6
                    dict_answer['image_7'] = answer.image_7
                    dict_answer['image_8'] = answer.image_8
                    dict_answer['image_9'] = answer.image_9
                    dict_answer['image_10'] = answer.image_10
                    dict_answer['image_11'] = answer.image_11
                    dict_answer['image_12'] = answer.image_12
                    dict_answer['image_13'] = answer.image_13
                    dict_answer['image_14'] = answer.image_14
                    dict_answer['image_15'] = answer.image_15

                return dict_answer
            except Exception as e:
                exit(e)

        else:
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                answer = response.choices[0].message.content

            except Exception as e:
                exit(e)

        return answer

    def text_infer(self, prompt=None, system_prompt=None):

        prompt = "What is the best football player in the world?" if prompt is None else prompt

        if system_prompt is not None:
            messages = [
                {"role": "system",
                 "content": system_prompt
                 },
            ]
        else:
            messages = []

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
        )
        response = chat_completion.choices[0].message
        # print(chat_completion.choices[0].message.content)

        return response


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_response(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    cnt_prompt_tokens = response.usage.prompt_tokens
    cnt_completion_tokens = response.usage.completion_tokens
    answer = response.choices[0].message.content
    return answer, cnt_prompt_tokens, cnt_completion_tokens


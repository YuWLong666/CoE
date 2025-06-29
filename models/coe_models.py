# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2024- 07- 08
Authors: Yu wenlong  and  DRAGON_501

*************************************Import***********************************"""

from models.internvl import internvl2 as internvl


"""**********************************Import***********************************"""
'''***************************************************************************'''


class ContextModel:

    def __init__(self, args, api_key="", text_refiner=None):
        self.args = args
        # you can use your own API key for OpenAI GPT or define it as your needs.
        if args.mllm_name == 'intern':
            self.captioner = internvl(args)
        else:
            raise ValueError('Invalid args.mllm_name')

        if 'intern' in args.mllm_name:
            self.llm = self.captioner.llm




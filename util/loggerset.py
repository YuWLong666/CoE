# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2023- 09- 22
Authors: Yu wenlong  and  DRAGON_501

*************************************Import***********************************"""
import os
import errno
import torch
import time
import logging
import shutil
import sys

"""**********************************Import***********************************"""
'''***************************************************************************'''

CNN_MODELS = ['resnet18','resnet34','resnet50','resnet101','resnet152',
              'rn18','rn34','rn50','rn101','rn152',
              'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
              'rn50_clip', 'rn50_mer', '', '',]


TRANSFORMER_MODELS = ['vit', 'bert', 'roberta', 'albert', 'electra', 'gpt2', 't5', '', '', '', '']

"""***************************************************************************"""
"""*********************************logger************************************"""


def makedirsExist(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
        else:
            raise


def basicset_logger(args, loggerpredix=''):
    args.document_root = os.getcwd()

    args.xai_method = ''
    if args.b_comp_LRP_CRP is True:
        args.xai_method += 'CRP'

    if args.resume is None:  # resume with full ckpt path

        args.train_type = 'xai'
        args.resume = None
        base_docu_name = '{}_{}_{}_{}_{}' \
                .format(args.train_type,
                        args.model_name, args.dataset, args.xai_method,
                        args.other)

        args.log_file_name = '{}_{}'.format(base_docu_name, args.time_all_start)

        args.douc_root = os.path.join(args.document_root, 'output')
        os.makedirs(args.douc_root, exist_ok=True)

        if args.gpu_location == 'auto':
            args.output_root = os.path.join('/root/autodl-tmp/', 'output')
            os.system('ln -s {} {}'.format(args.output_root, args.douc_root))
        else:
            args.output_root = args.douc_root

        if args.out_forder_type == 'o':
            if len(args.out_forder_name) == 0 or args.out_forder_name is None:
                pass
            else:
                args.output_root = os.path.join(args.output_root, args.out_forder_name)

        if args.model_name in CNN_MODELS:
            args.model_type = 'cnn'
        elif args.model_name in TRANSFORMER_MODELS:
                args.model_type = 'transfm'
        else:
            raise ValueError('model_name is not in CNN_MODELS or TRANSFORMER_MODELS')

        if args.other == 'debug':
            args.output_path = os.path.join(args.output_root, 'debug')
            args.writer_name = 'debug_{}_{}'.format(args.model_type, args.log_file_name)
        else:
            args.output_path = os.path.join(args.output_root, args.model_type)
            args.writer_name = args.log_file_name

        if args.out_forder_type == 'n':
            if len(args.out_forder_name) == 0 or args.out_forder_name is None:
                args.output_path = os.path.join(args.output_path, 'allothers')
            else:
                args.output_path = os.path.join(args.output_path, args.out_forder_name)

        args.output_dir = os.path.join(args.output_path, args.log_file_name)
        os.makedirs(args.output_dir, exist_ok=True)

        args.log_dir = args.output_dir

        mainpy_dir = os.path.join(args.document_root, 'main.py')
        shutil.copy(mainpy_dir, os.path.join(args.output_dir, 'copy_main.py'))

    else:
        args.train_type = 'resume'
        if args.resume.endswith('.pth'):
            args.output_root = os.path.dirname(args.resume)
            args.output_dir = os.path.join(args.output_root, os.path.basename(args.resume).rstrip('.pth'))
            base_docu_name = args.log_file_name = os.path.basename(args.resume).rstrip('.pth')
        else:
            args.output_root = args.resume
            args.output_dir = args.resume
            base_docu_name = args.log_file_name = os.path.basename(args.resume.rstrip('/'))
        args.output_path = args.output_dir
        args.log_dir = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)

        args.log_file_name = 'resume_{}_{}'.format(base_docu_name, args.time_all_start)

    if loggerpredix != '':
        log_file_log = f'{args.log_file_name}_{loggerpredix}.log'
    else:
        log_file_log = f'{args.log_file_name}.log'
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(os.path.join(args.log_dir, log_file_log))
    handler.setLevel(logging.DEBUG)
    han_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(han_formatter)
    logger.addHandler(handler)
    args.log_file_log = os.path.join(args.log_dir, log_file_log)
    return logger

# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2024- 09- 22
Authors: Yu wenlong  and  DRAGON_501

'''***************************************************************************'''
*************************************Import***********************************"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,6,7"

import torch

world_size = torch.cuda.device_count()

import util.misc as misc
from util.loggerset import basicset_logger
from util.plot_utils import *
from parser import get_args_parser
import cv2

from models import build_models
from data.data_proces import MakeDataset

from models.coe_models import ContextModel
from util.plot_utils import clear_plot, plt_show_save

from polysemantic_eval import main_polysemantic

from closeai import OpenAIGpt
import torch.nn as nn

from tqdm import tqdm
import time
import sys
import numpy as np
import random

import pandas as pd
import datetime
import json
import argparse

'''**********************************Import***********************************'''
'''***************************************************************************'''
torch.set_num_threads(1)
torch.set_printoptions(threshold=100, profile="short", sci_mode=False)
now = int(time.time())
timeArray = time.localtime(now)
Time = time.strftime("%Y%m%d_%H%M", timeArray)

ccid_length_model = {
                     'rn50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
                     'rn152': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
                     'vitb16_clip': {'layer0': 768, 'layer2': 768, 'layer9': 768, 'layer11': 768},
                     }


def main(args):

    logger = basicset_logger(args)
    args.logger = logger
    logger.info(f"\nCommand: {' '.join(sys.argv)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Resume
    b_comp_LRP_CRP = args.b_comp_LRP_CRP
    b_store_LRP_CRP = args.b_store_LRP_CRP

    ###############################################################################################################
    ###############################################################################################################
    # ************************************************************************************************
    # ******  Model Construction --- DVM ---  ******
    # *

    model, criterion, preprocess, tokenizer, msg = build_models(args)
    model = model.to(device)

    ccid_length_layer = ccid_length_model[args.model_name]

    model.eval()
    # ************************************************************************************************
    ###############################################################################################################

    ###############################################################################################################
    # ************************************************************************************************
    # ******  data configuration  ******
    # *
    import torchvision.transforms as T
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    args.dataset_name = args.dataset.split('-')[0]
    args.dataset_split = args.dataset.split('-')[1]

    if args.dataset_name == 'imagenet':
        args.data_dir = os.path.join(args.data_path, 'imagenet')
        IN_dataset_source = MakeDataset(args, transform=transform, dataset_split=args.dataset_split)

    if 'clip' in args.model_name:
        IN_dataset_source_split = IN_dataset_source.dataset_split
        in_classes_list = IN_dataset_source_split.classes
        in_classes_list = [f"A photo of a {label[0]}" for label in in_classes_list]
        clip_classes = tokenizer(in_classes_list).to(device)  # torch.Size([1000, 77])

        text_features = model.encode_text(clip_classes)  # torch.Size([1000, 1024])
        text_features /= text_features.norm(dim=-1, keepdim=True)

        class clip_ImageNet(nn.Module):

            def __init__(self, clip_model, text_features):
                super(clip_ImageNet, self).__init__()
                self.clip_model = clip_model.visual
                self.text_features = text_features
                self.text_features = nn.Parameter(self.text_features)

            def forward(self, x):

                # image_features = self.clip_model.encode_image(x)  # torch.Size([1, 1024])
                image_features = self.clip_model(x)  # torch.Size([1, 1024])
                image_features_norm = torch.norm(image_features, p=2, dim=-1, keepdim=True)
                image_features = image_features / image_features_norm

                text_probs = torch.matmul(image_features, self.text_features.permute(1, 0)) * 100  # torch.Size([1, 1000])
                text_probs = nn.Softmax(dim=-1)(text_probs)
                return text_probs

        clip_IN_model = clip_ImageNet(model, text_features)

    # ************************************************************************************************
    ###############################################################################################################

    ###############################################################################################################
    # ************************************************************************************************
    # **** acquire the index of current args.imgviz
    # *

    if args.imgviz_file is None:
        if args.imgviz is not None:
            args.imgvizs = [args.imgviz]
        else:
            args.imgvizs = []
    else:
        if '.txt' in args.imgviz_file:
            args.imgvizs = []
            with open(args.imgviz_file, 'r') as f:
                args.imgviz_all = f.readlines()
            for i_imgviz in args.imgviz_all:
                i_imgviz_file = i_imgviz.split(': ')[0]
                args.imgvizs.append(i_imgviz_file)

    test_img_num = 50
    b_first_img = True

    for i_img, args.imgviz in enumerate(tqdm(args.imgvizs, desc=f"Image_Samples_Local_Explanation")):

        if i_img >= test_img_num:
            exit(f'Already tested {test_img_num} imgs, exit.')

        transform_delete_gray = T.Compose([T.Resize([224, 224]),
                               T.ToTensor()])
        image = Image.open(args.imgviz).convert('RGB')
        sample = transform_delete_gray(image)  # torch.Size([1, 3, 224, 224])
        if sample.shape[0] == 1:
            test_img_num += 1
            continue
        else:
            pass

        b_find_idx = False
        img_idx = IN_dataset_source.select_idx(args.imgviz)
        if img_idx is not None:
            b_find_idx = True
            inputs_source, labels_source = IN_dataset_source.__getitem__(index=img_idx)
        else:
            inputs_source = Image.open(args.imgviz)
            inputs_source = transform(inputs_source).unsqueeze(0).to(device)
            labels_source = args.imgviz.split('/')[-1][:-4]
            img_idx = labels_source
            labels_source = [46]

        if b_find_idx is True:
            labels_source = torch.from_numpy(np.array(labels_source)).unsqueeze(0)
            inputs_source = inputs_source.unsqueeze(0)

        logger.info(f'Img {img_idx} Found')

        ###############################################################################################################
        ###############################################################################################################
        # ************************************************************************************************
        # Note:
        #   Get into the CRP method (as an example to compute the mapping function formulated in the main paper).
        #   You can change it into any other XAI method to compute any kind of concepts.
        #   The main function of CoE is utilized in this function.
        # *

        if b_comp_LRP_CRP is True:

            import torchvision.transforms as T
            from zennit.canonizers import SequentialMergeBatchNorm
            from zennit.composites import EpsilonPlusFlat
            from zennit.composites import Composite
            from crp_lrp.crp.cache import ImageCache
            from crp_lrp.crp.concepts import ChannelConcept
            from crp_lrp.crp.helper import get_layer_names, abs_norm
            from crp_lrp.crp.attribution import CondAttribution
            from crp_lrp.crp.visualization import FeatureVisualization
            from crp_lrp.crp.image import imgify, vis_opaque_img, plot_grid
            from crp_lrp.crp.helper import max_norm
            from torchvision.transforms.functional import gaussian_blur
            import zennit.image as zimage

            g_y_list = ['y']

            args.use_cuda = 'cuda' in args.device.type and torch.cuda.is_available()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            canonizers = [SequentialMergeBatchNorm()]

            if 'vit' not in args.model_name:
                composite = EpsilonPlusFlat(canonizers)
                concept_type = ['relevance', 'activation']
            else:
                composite = Composite()
                concept_type = ['activation']

            if 'clip' in args.model_name:
                attribution = CondAttribution(clip_IN_model, tokenizer, preprocess, g_y_list=g_y_list)
            else:
                attribution = CondAttribution(model, g_y_list=g_y_list)
            cc = ChannelConcept()

            # list the concept layers that you want to compute concept and CoE
            if 'rn' in args.model_name:

                if 'clip' in args.model_name:
                    model_crp = clip_IN_model
                else:
                    model_crp = model

                layer_names_prograess = {
                    # 'layer1': 'layer1.2.conv3',
                    # 'layer2': 'layer2.3.conv3',
                    # 'layer3': 'layer3.5.conv3',
                    'layer4': 'layer4.2.conv3',
                }
                layer_names = get_layer_names(model_crp, [torch.nn.Conv2d, torch.nn.Linear, torch.nn.AdaptiveAvgPool2d])
                layer_names_p = get_layer_names(model_crp, [torch.nn.Conv2d, torch.nn.Linear, torch.nn.AdaptiveAvgPool2d])
                layer_names_conv = get_layer_names(model_crp, [torch.nn.Conv2d])

                for i_layer_name in layer_names_p:
                    b_store_layer = False
                    for iiiiii in layer_names_prograess.values():
                        if iiiiii in i_layer_name:
                            b_store_layer = True
                            break
                        else:
                            b_store_layer = False
                    if b_store_layer is False:
                        if i_layer_name in layer_names:
                            layer_names.remove(i_layer_name)
                            if i_layer_name in layer_names_conv:
                                layer_names_conv.remove(i_layer_name)

                target_layers_base = {
                    # 'layer1': 'layer1.2.conv3',
                    # 'layer2': 'layer2.3.conv3',
                    # 'layer3': 'layer3.5.conv3',
                    'layer4': 'layer4.2.conv3',

                }

                if 'clip' in args.model_name:
                    target_layers_base = {k: f'clip_model.{v}' for k, v in target_layers_base.items()}

            elif 'vit' in args.model_name and 'clip' in args.model_name:

                model_crp = clip_IN_model
                layer_names_prograess = {
                    'layer0': 'clip_model.transformer.resblocks.0.mlp.c_proj',
                    # 'layer1': 'clip_model.transformer.resblocks.1.mlp.c_proj',
                    'layer2': 'clip_model.transformer.resblocks.2.mlp.c_proj',
                    # 'layer3': 'clip_model.transformer.resblocks.3.mlp.c_proj',
                    # 'layer4': 'clip_model.transformer.resblocks.4.mlp.c_proj',
                    # 'layer5': 'clip_model.transformer.resblocks.5.mlp.c_proj',
                    # 'layer6': 'clip_model.transformer.resblocks.6.mlp.c_proj',
                    # 'layer7': 'clip_model.transformer.resblocks.7.mlp.c_proj',
                    # 'layer8': 'clip_model.transformer.resblocks.8.mlp.c_proj',
                    'layer9': 'clip_model.transformer.resblocks.9.mlp.c_proj',
                    # 'layer10': 'clip_model.transformer.resblocks.10.mlp.c_proj',
                    'layer11': 'clip_model.transformer.resblocks.11.mlp.c_proj',
                }
                layer_names = get_layer_names(model_crp, [torch.nn.Linear])
                layer_names_p = get_layer_names(model_crp, [torch.nn.Linear])

                for i_layer_name in layer_names_p:
                    b_store_layer = False
                    for iiiiii in layer_names_prograess.values():
                        if iiiiii in i_layer_name:
                            b_store_layer = True
                            break
                        else:
                            b_store_layer = False
                    if b_store_layer is False:
                        if i_layer_name in layer_names:
                            layer_names.remove(i_layer_name)
                target_layers_base = {
                    'layer0': 'clip_model.transformer.resblocks.0.mlp.c_proj',
                    # 'layer1': 'clip_model.transformer.resblocks.1.mlp.c_proj',
                    'layer2': 'clip_model.transformer.resblocks.2.mlp.c_proj',
                    # 'layer3': 'clip_model.transformer.resblocks.3.mlp.c_proj',
                    # 'layer4': 'clip_model.transformer.resblocks.4.mlp.c_proj',
                    # 'layer5': 'clip_model.transformer.resblocks.5.mlp.c_proj',
                    # 'layer6': 'clip_model.transformer.resblocks.6.mlp.c_proj',
                    # 'layer7': 'clip_model.transformer.resblocks.7.mlp.c_proj',
                    # 'layer8': 'clip_model.transformer.resblocks.8.mlp.c_proj',
                    'layer9': 'clip_model.transformer.resblocks.9.mlp.c_proj',
                    # 'layer10': 'clip_model.transformer.resblocks.10.mlp.c_proj',
                    'layer11': 'clip_model.transformer.resblocks.11.mlp.c_proj',
                }

            else:
                raise ValueError('layer_names not defined for this model')

            # layer_map
            layer_map = {layer: cc for layer in layer_names}
            transform = T.Compose([T.Resize([224, 224]),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transform1 = T.Compose([T.Resize([224, 224]),
                                    T.ToTensor()])
            preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = Image.open(args.imgviz).convert('RGB')
            sample = transform(image).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])

            if args.ckpt_dir is not None and args.ckpt_dir != '':
                pth_name = args.ckpt_dir.split('/')[-1][:-4]
            else:
                pth_name = 'best'

            crp_base_dir = os.path.join(args.log_dir, f'crp_source_{pth_name}')
            dataset_crp = MakeDataset(args, tokenizer, model, preprocess, device, transform=transform1,
                                      dataset_split=args.dataset_split)
            if 'clip' in args.model_name:
                attribution.clip_classes = dataset_crp.clip_classes
                attribution.text_features = text_features
                dataset_crp.text_features = text_features

            crp_featstore_dir = f'{crp_base_dir}/featstore'
            os.makedirs(crp_featstore_dir, exist_ok=True)

            # set the path to store and test
            b_plot_imgviz = True
            if b_plot_imgviz is True:

                if 'clip' not in args.model_name:
                    sample_test = transform1(image).unsqueeze(0).to(device)
                    sample_test.requires_grad = False
                    out_test = model(sample_test)
                else:
                    img = preprocess(image).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])

                    image_features = model.encode_image(img)  # torch.Size([1, 1024])
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    out_test = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    # top_probs, top_labels = out_test.cpu().topk(1, dim=-1)

                pred_label = out_test.argmax(dim=1).item()
                if 'imagenet' in args.dataset:
                    pred_class = dataset_crp.dataset_split.classes[pred_label][0]
                    lable_name = dataset_crp.dataset_split.classes[labels_source[0]][0]

                # if you use your own dataset, customize here
                else:
                    raise ValueError('dataset not defined')

                sample_dir = f'{crp_base_dir}/img{img_idx}_label{labels_source[0]}_' \
                             f'{lable_name.replace(" ", "-")}__' \
                             f'pred{pred_label}_{pred_class.replace(" ", "-")}'
                os.makedirs(sample_dir, exist_ok=True)
                plt_show_save(image, path=sample_dir, name=f'org_224.jpg', b_show=False)

                sample.requires_grad = True

                img_orig = np.array(Image.open(args.imgviz).convert('RGB'))
                plt_show_save(img_orig, path=sample_dir, name=f'0.org.jpg')

                img_orig = np.array(Image.open(args.imgviz).convert('RGB').resize((224, 224)))
                plt_show_save(img_orig, path=sample_dir, name=f'0.org_224.jpg')

                img_orig = cv2.imread(args.imgviz)
                img_orig_resz = cv2.resize(img_orig, (224, 224))
                cv2.imwrite(os.path.join(sample_dir, f'0.org_cv2_resize.jpg'), img_orig_resz,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                img_resz_norm_orig = inputs_source[0].detach().clone().cpu().numpy()
                img_resz_norm_orig = (img_resz_norm_orig - np.min(img_resz_norm_orig)) / (
                        np.max(img_resz_norm_orig) - np.min(img_resz_norm_orig))
                img_resz_norm_orig = np.uint8(255 * img_resz_norm_orig)

                img_resz_orig = cv2.cvtColor(np.uint8(np.transpose(img_resz_norm_orig, (1, 2, 0))),
                                             cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(sample_dir, f'0.resize_norm_org.jpg'), img_resz_orig,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # ************************************************************************************************
            ###########################################################################################################

            ###########################################################################################################
            # ************************************************************************************************
            # 1. CoE------Global Explanation
            # *
            #
            # 1.1 Calculate the maximum activation sample index, activation value,
            # and neuron activation value of the relevance and activation of all channels,
            # and store them in the featstore folder.
            #
            # Each model to be explained only needs to be calculated once.
            # After being stored, it will no longer be calculated.
            #

            fv_all = {}
            saved_files = {}

            # args.b_force_store_LRP_CRP = True  # if you want to enforce the recomputation of CRP
            for i_y, g_y in enumerate(g_y_list):
                crp_store_ig_dir = f'{crp_featstore_dir}/{g_y}'
                has_stored = False
                if os.path.isdir(crp_store_ig_dir):
                    has_stored = True
                else:
                    os.makedirs(crp_store_ig_dir, exist_ok=True)
                fv_all[g_y] = FeatureVisualization(attribution, dataset_crp, layer_map, preprocess_fn=preprocessing,
                                                   path=crp_store_ig_dir, i_y=i_y)

                if (b_store_LRP_CRP is True and has_stored is False) or (args.b_force_store_LRP_CRP is True):
                    logger.info('Now compute and store all relevance values.')
                    saved_files[g_y] = fv_all[g_y].run(composite, 0, len(dataset_crp.dataset_split), 40, 200)
                else:
                    pass

            # ************************************************************************************************
            ###########################################################################################################

            ###########################################################################################################
            # ************************************************************************************************
            #
            # 1.2 Store 15 activation maps for each CCID and obtain the semantic entropy
            # and semantic description of polysemantics, as well as statistical information.
            # Acquire sem_ccids_atoms_probs_dict，
            # Read only at the first image.
            #

            # args.b_store_ccid_sem_imgs = True  # if you want to enforce the recomputation of this step
            if 'vit' in args.model_name:
                args.cctype = 'activation'

            if b_first_img is False or args.b_use_milan_dec is True:
                pass

            else:
                g_y = 'y'
                for j_cctype, cctype in enumerate(concept_type):

                    img_store_path = f'{crp_featstore_dir}/{cctype}/ccid_imgs/'
                    if os.path.exists(img_store_path):
                        pass
                    else:
                        args.b_store_ccid_sem_imgs = True

                    # ************************************************************************************************

                    if args.b_store_ccid_sem_imgs is True:

                        sem_desc_dict = {}
                        for ly_name, len_ccid_ly in ccid_length_layer.items():

                            if ly_name not in layer_names_prograess.keys():
                                continue

                            if args.force_layer is not None:
                                if ly_name != args.force_layer:
                                    continue

                            sem_desc_dict[ly_name] = {}

                            img_store_path_ly = os.path.join(img_store_path, ly_name)
                            os.makedirs(img_store_path_ly, exist_ok=True)

                            ly_layer = target_layers_base[ly_name]

                            ccids_ly = np.arange(len_ccid_ly)
                            ccids_ly = list(ccids_ly)

                            # You can parallelize this loop to speed up the computation.
                            # For example, you can use the "args.force_layer = 'layer4'" and "args.target_num=7"
                            # to specify the number of groups to be processed
                            # (i.e., the 8th group (channel id 1792 - 2047) of layer4 of RN152 model).

                            target_num = args.target_num
                            target_num = target_num * 256

                            for i_c, ccid in tqdm(enumerate(ccids_ly), total=len(ccids_ly),
                                                  desc=f'CC Img Patch: {cctype}_{ly_name}'):

                                if os.path.exists(os.path.join(img_store_path_ly, f'{ccid}_14.JPEG')):
                                    continue
                                if args.force_layer is not None:
                                    if target_num <= i_c < target_num + 256:
                                        pass
                                    else:
                                        continue

                                ref_c, d_indices, d_labels_all = fv_all[g_y].get_max_reference(concept_ids=[ccid],
                                                                                               layer_name=ly_layer,
                                                                                               mode=cctype,
                                                                                               r_range=(0, 15),
                                                                                               composite=composite,
                                                                                               plot_fn=None)
                                B, C, H, W = ref_c[ccid][0].shape
                                if 'vit' not in args.model_name:
                                    ccid_imgs = ref_c[ccid][0]  # [15, 3, 224, 224]
                                    ccid_masks = ref_c[ccid][1].view(B, 1, H, W)  # [15, 224, 224]

                                    ccid_masked_imgs = ccid_masks.repeat(1, 3, 1, 1) * ccid_imgs

                                    for ii, i_masked_img in enumerate(ccid_masked_imgs):

                                        ii_ccid_img = ccid_imgs[ii]
                                        ii_ccid_mask = ccid_masks[ii]

                                        filtered_heat = max_norm(gaussian_blur(ii_ccid_mask, kernel_size=19)[0])
                                        vis_mask = filtered_heat > 0.2
                                        inv_mask = ~vis_mask
                                        ii_ccid_masked_img = ii_ccid_img * vis_mask + ii_ccid_img * inv_mask * 0
                                        ii_ccid_masked_img = zimage.imgify(ii_ccid_masked_img.detach().cpu())

                                        img_name = f'{ccid}_{ii}.JPEG'

                                        plt_show_save(ii_ccid_masked_img, path=img_store_path_ly, name=img_name, dpi=300,
                                                      b_show=False)

                                else:
                                    ccid_imgs = ref_c[ccid][0]  # [15, 3, 224, 224]

                                    for ii, i_masked_img in enumerate(ccid_imgs):

                                        ii_ccid_img = ccid_imgs[ii]
                                        img_name = f'{ccid}_{ii}.JPEG'

                                        plt_show_save(ii_ccid_img, path=img_store_path_ly, name=img_name,
                                                      dpi=300,
                                                      b_show=False)

                img_store_path = f'{crp_featstore_dir}/{args.cctype}/ccid_imgs'
                ccid_desc_path = f'{crp_featstore_dir}/{args.cctype}/ccid_desc'

                # ************************************************************************************************
                #
                # Key point of our paper:
                # Calculate the semantic entropy of each concept,
                # automatically construct the global concept explanation dataset
                # and calculate their concept polysemanticity entropy.
                #
                sem_desc_dict, \
                semantic_entropys_dict, \
                static_entropy_all_dict, \
                static_entropy_avg_dict, \
                sem_ccids_atoms_probs_dict = \
                    main_polysemantic(args, img_store_paths=img_store_path, ccid_desc_path=ccid_desc_path)

            # ************************************************************************************************
            ###########################################################################################################

            ###########################################################################################################
            # ************************************************************************************************

            target_layers = {'y': target_layers_base}

            #
            ###########################################################################################################
            # ************************************************************************************************
            # 2. CoE ------Local Explanation
            #

            toprel_ccid_dict = {}
            toprel_value_dict = {}
            sorted_ccid_dict = {}
            sorted_value_dict = {}
            b_norm = args.b_norm
            for j_cctype, cctype in enumerate(concept_type):
                for i_y, g_y in enumerate(g_y_list):

                    conditions = [{g_y: [torch.tensor(pred_label)]}]

                    heatmaps, activations, relevances, predictions = attribution(sample, conditions, composite)

                    img = imgify(heatmaps, symmetric=True, grid=(1, len(heatmaps)))

                    condition_str = str(conditions).replace(':', '-')
                    plt_show_save(img, b_show=False, path=sample_dir,
                                  name=f'2.img{img_idx}_cond{condition_str}_heatmap.jpg',
                                  title=f'img{img_idx}_cond{condition_str}_heat')

                    if isinstance(target_layers, dict):
                        target_layers_gy = {**target_layers[g_y], **target_layers_base}
                    else:
                        target_layers_gy = target_layers_base

                    heatmaps, activations, relevances, predictions = attribution(sample, conditions, composite,
                                                                                 record_layer=layer_names)

                    for ly_name, ly_layer in target_layers_gy.items():
                        if cctype == 'relevance':
                            rel_c = cc.attribute(relevances[ly_layer], abs_norm=b_norm)
                        else:
                            rel_c = cc.attribute(activations[ly_layer], abs_norm=b_norm)

                        rel_values, concept_ids = rel_c.topk(10, dim=1)
                        rel_values_sort, concept_ids_sort = torch.sort(rel_c, descending=True, dim=1)

                        toprel_ccid_dict[f'{cctype}_{g_y}_{ly_name}'] = np.array(concept_ids.detach().cpu()).flatten()
                        toprel_value_dict[f'{cctype}_{g_y}_{ly_name}'] = np.array(rel_values.detach().cpu()).flatten()
                        sorted_ccid_dict[f'{cctype}_{g_y}_{ly_name}'] = np.array(
                            concept_ids_sort.detach().cpu()).flatten()
                        sorted_value_dict[f'{cctype}_{g_y}_{ly_name}'] = np.array(
                            rel_values_sort.detach().cpu()).flatten()

            sorted_ccid_pd = pd.DataFrame.from_dict(sorted_ccid_dict, orient='index').transpose()
            sorted_value_pd = pd.DataFrame.from_dict(sorted_value_dict, orient='index').transpose()

            sorted_ccid_pd.to_csv(f'{sample_dir}/2.img{img_idx}_cond{condition_str}_all_sorted_ccid.csv', mode='w',
                                  encoding='utf-8')
            sorted_value_pd.to_csv(
                f'{sample_dir}/2.img{img_idx}_cond{condition_str}_all_sorted_ccid_value_norm{b_norm}.csv', mode='w',
                encoding='utf-8')

            # ************************************************************************************************
            ##########################################################################################################

            ##########################################################################################################
            # ************************************************************************************************
            # ******  CoE Model Construction  ******
            #

            img_context_path = f'{sample_dir}/img_context.json'

            b_use_context = True

            #
            ###########################################################################################################
            #
            # Define the model only for the first time.
            # You can define these functions as your needs.
            #
            if b_first_img is True:

                if 'intern' in args.mllm_name:  # We use InternVL as our context model. You can use other models.
                    coe_context_model = ContextModel(args)

                if 'gpt' in args.atom_select_llm:
                    coe_atom_select_model = OpenAIGpt(model_name=args.atom_select_llm, b_format=True, temperature=0.1)

                if 'gpt' in args.coe_final_llm:
                    coe_localxai_model = OpenAIGpt(model_name=args.coe_final_llm, b_format=True, temperature=0.2)

                # logger.info(f'Finish load or construct CoE model.')
            else:
                pass

            b_first_img = False

            #
            ###########################################################################################################
            # Caption context of the input image (local explanation)
            #
            if b_use_context is True:
                if os.path.exists(img_context_path) is False:

                    prompt_context = '<image>\nPlease describe the image. No more than 40 words. ' \
                               'You should describe some attributes of each object in the given image, ' \
                               'such as color, shape, size, etc.' \
                               'Keep your description concise and veritable, rather than imagination or exaggeration.'

                    if args.mllm_name == 'intern':
                        image_context = coe_context_model.captioner.inference(args.imgviz, b_single=True, prompt=prompt_context)
                        with open(img_context_path, 'w') as f:
                            json.dump(image_context, f)
                    else:
                        pass

                    # logger.info(f'Whole image_context: {image_context[0]}')

                elif b_use_context is True:
                    if args.mllm_name == 'intern':
                        with open(img_context_path, 'r') as f:
                            image_context = json.load(f)
                    else:
                        pass
            else:
                image_context = None

            #
            # ************************************************************************************************
            ###########################################################################################################
            ###########################################################################################################
            # ************************************************************************************************
            #
            # Find the top-k activated concepts for each concept, and then
            # determine whether the correlation is greater than the threshold.
            # If so, record the concept atom set of that concept.
            #
            aerfa_threshold = 0.7
            ex_layers = list(target_layers_base.keys())

            key_name = 'sem_probs_cluster_dict'
            entropy_key_name = 'sem_entropy_naive_pad_log'

            i_y = 0
            g_y = g_y_list[i_y]

            cctype = args.cctype
            if isinstance(target_layers, dict):
                target_layers_gy = {**target_layers[g_y], **target_layers_base}
            else:
                target_layers_gy = target_layers_base

            cc_dec_path = os.path.join(sample_dir, 'concept_description')
            os.makedirs(cc_dec_path, exist_ok=True)
            cc_dec_milan_path = os.path.join(sample_dir, 'concept_description_milan')
            os.makedirs(cc_dec_milan_path, exist_ok=True)
            captions_layer = {}
            cap_prob_entropy = {}

            b_clip_dissect = False

            captions_layer_path = os.path.join(sample_dir, 'key_captions_layer.json')
            cap_prob_entropy_path = os.path.join(sample_dir, 'cap_prob_entropy.json')

            # ************************************************************************************************
            # ************************************************************************************************
            #
            # Start reading the "caption_layer.json" and "cap_prob_entropy.json" files.
            # If they do not exist, then start to re-acquire them.

            if os.path.exists(captions_layer_path) is False or os.path.exists(cap_prob_entropy_path) is False:
                #
                channal_captions_all_dict = {}
                for ly_name, ly_layer in target_layers_gy.items():
                    if ly_name in ex_layers:
                        ccname = f'{cctype}_{g_y}_{ly_name}'

                        cc_dec_layer_path = os.path.join(cc_dec_path, ly_name)
                        os.makedirs(cc_dec_layer_path, exist_ok=True)
                        cc_dec_layer_milan_path = os.path.join(cc_dec_milan_path, ly_name)
                        os.makedirs(cc_dec_layer_milan_path, exist_ok=True)

                        captions_layer[ccname] = {}
                        cap_prob_entropy[ccname] = {}
                        toprel_ccid_layer = toprel_ccid_dict[ccname]

                        if ccname not in channal_captions_all_dict.keys():
                            channal_captions_all_dict[ccname] = {}

                        toprel_ccid_value_layer = toprel_value_dict[ccname] / max(toprel_value_dict[ccname])

                        for i_ccid, ccid in enumerate(toprel_ccid_layer):
                            ccid_value = toprel_ccid_value_layer[int(i_ccid)]
                            if ccid_value < aerfa_threshold:
                                continue
                            ccid_name = f'{i_ccid}-{ccid}'
                            cap_prob_entropy[ccname][ccid_name] = {}

                            topccid_content = sem_ccids_atoms_probs_dict[ly_name][str(ccid)][key_name]
                            topccid_entropy = semantic_entropys_dict[ly_name][str(ccid)][entropy_key_name]
                            topccid_annots = sem_ccids_atoms_probs_dict[ly_name][str(ccid)][key_name].keys()

                            captions_layer[ccname][ccid_name] = list(topccid_annots)
                            cap_prob_entropy[ccname][ccid_name]['content'] = topccid_content
                            cap_prob_entropy[ccname][ccid_name]['entropy'] = topccid_entropy
                            channal_captions_all_dict[ccname][ccid_name] = list(topccid_annots)

                with open(captions_layer_path, 'w') as f:
                    json.dump(captions_layer, f)
                with open(cap_prob_entropy_path, 'w') as f:
                    json.dump(cap_prob_entropy, f)
            else:
                with open(captions_layer_path, 'r') as f:
                    captions_layer = json.load(f)
                with open(cap_prob_entropy_path, 'r') as f:
                    cap_prob_entropy = json.load(f)

            #
            # ************************************************************************************************
            ###########################################################################################################
            ###########################################################################################################
            # ************************************************************************************************
            # Read the file "coe_info_dict.json". If it does not exist, create it.
            # coe_atom_select_model
            # Based on the relevance of topk, accumulate the captions,
            # iccid, ccid, ccid_value, ccid_cap_dict, and ccid_value_dict for each layer.
            #
            layer_captions = {}
            layer_cap_iccid = {}
            layer_cap_ccid = {}
            layer_cap_ccid_value = {}
            layer_cap_ccid_cap_dict = {}
            layer_cap_ccid_value_dict = {}
            layer_cap_ccid_prob_dict = {}
            layer_cap_ccid_entropy_dict = {}
            layer_cap_ccid_cap_prob_dict = {}

            if args.atom_select == 'random':
                coe_info_path = os.path.join(sample_dir, 'coe_info_dict_random.json')

            elif args.atom_select == 'llm':
                coe_info_path = os.path.join(sample_dir, 'coe_info_dict_llm.json')

            else:
                raise NotImplementedError

            if os.path.exists(coe_info_path) is False:

                coe_info_dict = {}
                for ccname in captions_layer.keys():
                    cc_type, g_y, ly_name = ccname.split('_')
                    layer_captions[f'{ly_name}'] = []
                    layer_cap_iccid[f'{ly_name}'] = []
                    layer_cap_ccid[f'{ly_name}'] = []
                    layer_cap_ccid_value[f'{ly_name}'] = []

                    layer_cap_ccid_cap_dict[f'{ly_name}'] = {}
                    layer_cap_ccid_value_dict[f'{ly_name}'] = {}
                    layer_cap_ccid_prob_dict[f'{ly_name}'] = {}
                    layer_cap_ccid_entropy_dict[f'{ly_name}'] = {}
                    layer_cap_ccid_cap_prob_dict[f'{ly_name}'] = {}

                    toprel_ccid_value_layer = toprel_value_dict[ccname] / max(toprel_value_dict[ccname])

                    for i_ccid_ccid, vc_cc_captions in captions_layer[ccname].items():
                        i_ccid, ccid = i_ccid_ccid.split('-')
                        ccid_value = toprel_ccid_value_layer[int(i_ccid)]
                        if ccid_value < aerfa_threshold:
                            continue

                        if args.atom_select == 'random':
                            caption_select = random.choice(vc_cc_captions)
                            cc_atom_prob_dict = None
                            cc_atom_entropy = cap_prob_entropy[ccname][i_ccid_ccid]['entropy']

                        elif args.atom_select == 'llm':
                            cc_atom_set = vc_cc_captions
                            if isinstance(image_context, str):
                                pass
                            elif isinstance(image_context, list):
                                image_context = image_context[0]

                            cc_atom_prob_dict = None
                            cc_atom_entropy = cap_prob_entropy[ccname][i_ccid_ccid]['entropy']

                            ask_prompt = "Given a set of concept atoms, " \
                                         "please filter one atom from this set that is most relevant to the given description.\n" \
                                         "The set of concept atoms is: {cc_atom_set}.\n" \
                                         "The description is: '{image_context}' \n" \
                                         "Your responce should be only a single concept atom selected from the given set. " \
                                         "Now please provide your answer: "
                            ask_prompt = ask_prompt.format(cc_atom_set=cc_atom_set, image_context=image_context)

                            if 'gpt' in args.atom_select_llm:
                                caption_select = coe_atom_select_model.text_infer(ask_prompt).content.strip()

                            for i_atom, atom in enumerate(cc_atom_set):
                                if atom in caption_select:
                                    caption_select_sure = atom
                                    break
                                else:
                                    caption_select_sure = random.choice(vc_cc_captions)
                            caption_select = caption_select_sure

                        else:
                            raise NotImplementedError

                        layer_captions[f'{ly_name}'].append(caption_select)
                        layer_cap_ccid[f'{ly_name}'].append(ccid)
                        layer_cap_ccid_value[f'{ly_name}'].append(str(round(ccid_value, 2)))
                        layer_cap_iccid[f'{ly_name}'].append(i_ccid)

                        layer_cap_ccid_cap_dict[f'{ly_name}'][f'{ccid}'] = caption_select
                        layer_cap_ccid_value_dict[f'{ly_name}'][f'{ccid}'] = str(round(ccid_value, 2))

                        layer_cap_ccid_cap_prob_dict[f'{ly_name}'][f'{ccid}'] = cc_atom_prob_dict if cc_atom_prob_dict is not None else None
                        layer_cap_ccid_prob_dict[f'{ly_name}'][f'{ccid}'] = cc_atom_prob_dict[caption_select] if cc_atom_prob_dict is not None else None
                        layer_cap_ccid_entropy_dict[f'{ly_name}'][f'{ccid}'] = cc_atom_entropy if cc_atom_entropy is not None else None

                coe_info_dict['pred_class'] = pred_class
                coe_info_dict['lable_name'] = lable_name
                coe_info_dict['image_context'] = image_context
                coe_info_dict['layer_captions'] = layer_captions
                coe_info_dict['layer_cap_iccid'] = layer_cap_iccid
                coe_info_dict['layer_cap_ccid'] = layer_cap_ccid
                coe_info_dict['layer_cap_ccid_value'] = layer_cap_ccid_value
                coe_info_dict['layer_cap_ccid_cap_dict'] = layer_cap_ccid_cap_dict
                coe_info_dict['layer_cap_ccid_value_dict'] = layer_cap_ccid_value_dict

                coe_info_dict['layer_cap_ccid_cap_prob_dict'] = layer_cap_ccid_cap_prob_dict
                coe_info_dict['layer_cap_ccid_prob_dict'] = layer_cap_ccid_prob_dict
                coe_info_dict['layer_cap_ccid_entropy_dict'] = layer_cap_ccid_entropy_dict

                with open(coe_info_path, 'w') as f:
                    json.dump(coe_info_dict, f)

            else:
                with open(coe_info_path, 'r') as f:
                    coe_info_dict = json.load(f)
                pred_class = coe_info_dict['pred_class']
                lable_name = coe_info_dict['lable_name']
                image_context = coe_info_dict['image_context']

                layer_cap_ccid_cap_dict = coe_info_dict['layer_cap_ccid_cap_dict']
                layer_cap_ccid_value_dict = coe_info_dict['layer_cap_ccid_value_dict']

            #
            # ************************************************************************************************
            ###########################################################################################################
            ###########################################################################################################
            # ************************************************************************************************
            # 2. CoE------Local Explanation
            #

            if args.atom_select == 'random':
                final_response_path = os.path.join(sample_dir, 'final_response_random.json')

            elif args.atom_select == 'llm':
                final_response_path = os.path.join(sample_dir, 'final_response_llm.json')

            if os.path.exists(final_response_path) is not True:

                final_system_prompt = "You are an intelligent deep learning model explainer and " \
                                      "you are now explaining the decision predicted by a deep vision identification model. "\

                final_prompt_template = \
                    "Given a prediction of a deep vision model and its prediction path formulated in the format of a concept circuit, " \
                    "you should first judge whether the model prediction is correct or incorrect and " \
                    "then give the reason why the prediction is correct or incorrect based on the following information (A,B,C,D,E). " \
                    "You should generate an aggregated and rigorous paragraph based on the given information (A,B,C,D,E) rather than imagination. " \
 \
                    "The information (A,B,C,D,E) includes the following: \n" \
                    "A) The Model Prediction of the Input Image: {prediction}." \
                    "B) The Ground Truth Label of the Input Image: {target_label}." \
                    "C) The Caption of the Input Image: {image_caption}." \
                    "D) The decision path of the vision model which is the structured relevant concept descriptions from the shallowest layer to the deepest layer of the model: " \
                    "{layer_ccid_decription}." \
                    "E) Structured relevant concept relevance value from the shallowest layer to the deepest layer of the model: " \
                    "{layer_ccid_rel_value}.\n" \
 \
                    "There are some rules for your response: \n" \
                    "Less than 20 sentences. " \
                    "Select the relevant properties of the image from the given information and describe the possible relation among them. " \
                    "Do not describe any individual letter. " \
                    "You need first to judge whether the prediction is correct or incorrect based on the given information. " \
                    "The prediction of the model can only be correct or incorrect, not both. " \
 \
                    "You are given a positive and a negative example to guide your response. " \
                    "Your response should obey the following example templates exactly. \n" \
 \
                    "A positive example of predicting correctly is given information as:\n" \
                    "A) The Model Prediction of the Input Image: dog. " \
                    "B) The Ground Truth Label of the Input Image: dog. " \
                    "C) The Caption of the Input Image: a spotty dog standing in the green grass, looking up at the sky. " \
                    "D) Structured relevant concept descriptions from the shallowest layer to the deepest layer of the model: " \
                    "{{layer1: {{5: colour, 7: stripe, 154: spot, 20: middle of the image, 6: edge of objects}}, " \
                    "layer2: {{224: green, 1028: dog, 1000: head, 90: sky, 85: grass}}}}. " \
                    "E) Structured relevant concept relevance value from the shallowest layer to the deepest layer of the model: " \
                    "{{layer1: {{5: 1, 7: 0.85, 154: 0.76, 20: 0.55, 6: 0.52}}," \
                    "layer2: {{224: 1, 1028: 0.96, 1000: 0.73, 90: 0.66, 85: 0.32}}}}.  \n" \
 \
                    "You can use the following output template to give the reason for the correct prediction of the model: " \
                    "The model outputs a correct result dog. " \
                    "Specifically, in the shallowest layer 1 of the model, " \
                    "channel 5 with a relevance value of 1.0 describes the concept of colour, " \
                    "channel 7 with a relevance value of 0.85 describes the concept of stripe, " \
                    "channel 154 with a relevance value of 0.76 describes the concept of spot, " \
                    "channel 20 with a relevance value of 0.55 may want to locate a position in the middle of the image, " \
                    "and channel 6 with a relevance value of 0.52 may want to detect an edge of objects. " \
                    "In the deeper layer 2 of the model, " \
                    "channel 224 with a relevance value of 1.0 describes the concept of green which may be related to the colour detector channel 5 in the first layer, " \
                    "channel 1028 with a relevance value of 0.96 describes the concept of dog, " \
                    "channel 1000 with a relevance value of 0.73 describes the concept head, " \
                    "channel 90 with a relevance value of 0.66 describes the concept sky, " \
                    "and channel 85 with a relevance value of 0.32 describes the concept grass. " \
                    "We can see that all of these concepts are related to the dog object in the image. " \
 \
                    "Therefore, the model outputs a correct result dog. \n" \
                    "Besides, an incorrect prediction example is given information as: \n" \
                    "A) The Model Prediction of the Input Image: dog. " \
                    "B) The Ground Truth Label of the Input Image: eagle. " \
                    "C) The Caption of the Input Image: on the vast grassland, an eagle flies in the blue sky. " \
                    "D) Structured relevant concept descriptions from the shallowest layer to the deepest layer of the model: " \
                    "{{layer1: {{5: colour, 7: stripe, 154: spot, 20: middle of the image, 6: edge of objects}}, " \
                    "layer2: {{224: green, 1028: dog, 1000: head, 90: sky, 85: grass}}}}. " \
                    "E) Structured relevant concept relevance value from the shallowest layer to the deepest layer of the model: " \
                    "{{layer1: {{5: 1, 7: 0.85, 154: 0.76, 20: 0.55, 6: 0.52}}, " \
                    "layer2: {{224: 1, 1028: 0.96, 1000: 0.73, 90: 0.66, 85: 0.32}}}}. \n" \
                    "You can use the following output template to describe why the model predicts an incorrect result: " \
                    "The model outputs an incorrect result dog instead of the correct label eagle. " \
                    "Specifically, in the shallowest layer 1 of the model, " \
                    "channel 5 with a relevance value of 1.0 describes the concept of colour, " \
                    "channel 7 with a relevance value of 0.85 describes the concept of stripe, " \
                    "channel 154 with a relevance value of 0.76 describes the concept of spot, " \
                    "channel 20 with a relevance value of 0.55 may want to locate a position in the middle of the image, " \
                    "and channel 6 with a relevance value of 0.52 may want to detect an edge of objects." \
                    "In the deeper layer 2 of the model, " \
                    "channel 224 with a relevance value of 1.0 describes the concept of green which may be related to the colour detector channel 5 in the first layer, " \
                    "channel 1028 with a relevance value of 0.96 describes the concept of dog, " \
                    "channel 1000 with a relevance value of 0.73 describes the concept of head, " \
                    "channel 90 with a relevance value of 0.66 describes the concept of sky, " \
                    "and channel 85 with a relevance value of 0.32 describes the concept of grass. " \
                    "While the model predicts green grass and the sky, it does not predict the concepts associated with eagles. " \
                    "At the same time, concepts related to dogs in the model are activated, " \
                    "so the model gives wrong predictions about dogs. " \
                    "Therefore, the model outputs an incorrect result dog. \n" \
                    "Your inference process should follow these steps: " \
                    "Step 1, Based on information A), which is the model's prediction, and information B, which is the ground truth label of the input image, " \
                    "you need to first determine whether the two are semantically equivalent. " \
                    "If they are semantically equivalent, then the model's prediction is considered correct. " \
                    "If the prediction and the label are not semantically equivalent, it is considered an incorrect prediction." \
                    "Step 2, Based on the judgment in Step 1 and the given information C, D, and E, which include the caption of the input image, " \
                    "the vision model's decision path and the concept information at each node along the path, and the concept relevance values at each node, " \
                    "you need to explain why the model arrived at this correct or incorrect prediction. " \
                    "Analyze the decision process by examining each concept in the decision path to determine how they contributed to the final outcome. \n" \
                    "Now, please provide your response: "

                final_prompt = final_prompt_template.format(**{
                    "prediction": pred_class,
                    "target_label": lable_name,
                    "image_caption": image_context,
                    "layer_ccid_decription": layer_cap_ccid_cap_dict,
                    "layer_ccid_rel_value": layer_cap_ccid_value_dict,
                })

                if 'gpt' in args.coe_final_llm:
                    final_response = coe_localxai_model.text_infer(final_prompt,
                                                                system_prompt=final_system_prompt).content.strip()
                else:
                    raiser = ValueError(f'Invalid coe_final_llm: {args.coe_final_llm}')

                with open(final_response_path, 'w') as f:
                    json.dump({'final_prompt': final_prompt,
                               'final_response': final_response}, f)

            else:
                with open(final_response_path, 'r') as f:
                    final_response_dict = json.load(f)
                final_prompt = final_response_dict['final_prompt']
                final_response = final_response_dict['final_response']

            b_first_img = False
            continue

    del args.logger, logger
    b = None
    all_domain_csv_dir = None
    return b, all_domain_csv_dir


if __name__ == '__main__':
    start_time_all_all = time.time()
    parser = argparse.ArgumentParser('main', parents=[get_args_parser(stages=['generate', 'compute'])])
    args = parser.parse_args()

    document_root = os.getcwd()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_ckpt_root = args.main_ckpt_root
    resume_list = []
    args.time_all_start = time.strftime('%m%d%H%M')

    args.imgviz = './dataset/imagenet/val/n02089078/ILSVRC2012_val_00006401.JPEG'
    if 'imagenet' in args.dataset:
        args.imgviz_file = './coe_eval_results/RN152_wrong_right_samples_selected.txt'
    else:
        args.imgviz_file = None

    args.imgviz_file = None

    if args.resume == 'rn50':
        args.model_name = args.resume
        # args.resume = './output/pretrained/cnn/xai_rn50_imagenet-val_CRP_ckptdownload'
        args.resume = './output/viz/rn50'

    elif args.resume == 'rn152' and 'imagenet' in args.dataset:
        args.model_name = args.resume
        args.resume = './output/pretrained/cnn/xai_rn152_imagenet-val_CRP_ckptdownload'
        args.resume = './output/git-coe/cnn/xai_rn152_imagenet-val_CRP_ckptdownload'
        args.resume = './output/git-coe-test/cnn/xai_rn152_imagenet-val_CRP_ckptdownload'

    elif args.resume == 'rn50_clip':
        args.model_name = args.resume
        args.resume = './output/pretrained/cnn/xai_rn50_clip_imagenet-val_CRP_openai'

    elif args.resume == 'vitb16_clip':
        args.model_name = args.resume
        args.resume = './output/pretrained/vit/xai_vit_b16_clip_imagenet-val_CRP_ckptdownload'

    elif args.resume is None or args.resume == 'None' or args.resume == '':
        args.resume = None
    else:
        raise ValueError('Invalid resume model name: {}'.format(args.resume))

    ###############################################################
    # main
    ###############################################################
    result_pd, all_domain_csv_dir = main(args)

    print('all finished')

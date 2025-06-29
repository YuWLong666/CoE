# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2024- 10- 29
Authors: Yu wenlong  and  DRAGON_501

*************************************Import***********************************"""
import torch

world_size = torch.cuda.device_count()
from torch.utils.data import DataLoader

import argparse
import json
import pandas as pd

from tqdm import tqdm

from util.plot_utils import *
from parser import get_args_parser

from models import build_models
from data.data_proces import MakeDataset
import torchvision.transforms as T

from closeai import OpenAIGpt

"""**********************************Import***********************************"""
'''***************************************************************************'''

parser = argparse.ArgumentParser('CoE-eval', parents=[get_args_parser()])

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.imgviz = './dataset/imagenet/val/n02089078/ILSVRC2012_val_00006401.JPEG'
args.imgviz_file = None

if args.resume == 'rn50':
    args.model_name = args.resume
    # args.resume = './output/pretrained/cnn/xai_rn50_imagenet-val_CRP_ckptdownload'
    args.resume = './output/viz/rn50'

elif args.resume == 'rn152' and 'imagenet' in args.dataset:
    args.model_name = args.resume
    args.resume = './output/git-coe/cnn/xai_rn152_imagenet-val_CRP_ckptdownload'

elif args.resume == 'rn50_clip':
    args.model_name = args.resume
    args.resume = './output/pretrained/cnn/xai_rn50_clip_imagenet-val_CRP_openai'

elif args.resume == 'vitb16_clip':
    args.model_name = args.resume
    args.resume = './output/pretrained/vit/xai_vit_b16_clip_imagenet-val_CRP_ckptdownload'
elif args.resume == 'vitb16':
    args.model_name = args.resume
    args.resume = './output/pretrained/vit/xai_vit_b16_imagenet-val_CRP_ckptdownload'

elif args.resume is None or args.resume == 'None' or args.resume == '':
    args.resume = None
else:
    raise ValueError('Invalid resume model name: {}'.format(args.resume))

###############################################################################################################
###############################################################################################################
# ************************************************************************************************
# ******  Model Construction --- DVM ---  ******
# *
model, criterion, preprocess, tokenizer, msg = build_models(args)
model = model.to(device)
if criterion is not None:
    criterion = criterion.to(device)
model.eval()

###############################################################################################################
# ************************************************************************************************
# ******  data configuration  ******
# *

coe_eval_llm_model = OpenAIGpt(model_name=args.coe_eval_llm, b_format=True, format_func_name='cpe_eval',
                               temperature=0.1)

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

args.dataset_name = args.dataset.split('-')[0]
args.dataset_split = args.dataset.split('-')[1]
transform1 = T.Compose([T.Resize([224, 224]),
                        T.ToTensor()])

if args.dataset_name == 'imagenet':
    args.data_dir = os.path.join(args.data_path, 'imagenet')
    IN_dataset_source = MakeDataset(args, transform=transform, dataset_split=args.dataset_split)
    IN_dataload_source = torch.utils.data.DataLoader(IN_dataset_source, batch_size=1, shuffle=False, num_workers=2)

dataset_crp = MakeDataset(args, tokenizer, model, preprocess, device, transform=transform1,
                                      dataset_split=args.dataset_split)
# ************************************************************************************************
###############################################################################################################

###############################################################################################################
# ************************************************************************************************

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

args.log_dir = args.resume

CoE_eval_all_dict = {}
CoE_eval_total_score_dict = {}
CoE_eval_score1_dict = {}
CoE_eval_score2_dict = {}
CoE_eval_score3_dict = {}

eval_methods = [args.atom_select]

if args.ckpt_dir is not None and args.ckpt_dir != '':
    pth_name = args.ckpt_dir.split('/')[-1][:-4]
else:
    pth_name = 'best'
crp_base_dir = os.path.join(args.log_dir, f'crp_source_{pth_name}')
crp_featstore_dir = f'{crp_base_dir}/featstore'

answer_all_file = os.path.join(crp_base_dir, f'coe_eval_answer_allsamples_{args.atom_select}.json')
answer_total_score_file = os.path.join(crp_base_dir, f'coe_eval_total_score_allsamples_{args.atom_select}.json')
answer_score_1_file = os.path.join(crp_base_dir, f'coe_eval_score_1_allsamples_{args.atom_select}.json')
answer_score_2_file = os.path.join(crp_base_dir, f'coe_eval_score_2_allsamples_{args.atom_select}.json')
answer_score_3_file = os.path.join(crp_base_dir, f'coe_eval_score_3_allsamples_{args.atom_select}.json')

b_re_eval = False
if os.path.exists(answer_all_file) is True and b_re_eval is False:

    with open(answer_all_file, 'r') as f:
        CoE_eval_all_dict = json.load(f)
    with open(answer_total_score_file, 'r') as f:
        CoE_eval_total_score_dict = json.load(f)
    with open(answer_score_1_file, 'r') as f:
        CoE_eval_score1_dict = json.load(f)
    with open(answer_score_2_file, 'r') as f:
        CoE_eval_score2_dict = json.load(f)
    with open(answer_score_3_file, 'r') as f:
        CoE_eval_score3_dict = json.load(f)

else:
    test_img_num = 50
    test_order = 80
    eval_num = 0
    for i_img, args.imgviz in enumerate(tqdm(args.imgvizs, desc='ImgViz')):

        if eval_num >= test_img_num:
            break

        transform_delete_gray = T.Compose([T.Resize([224, 224]),
                               T.ToTensor()])
        image = Image.open(args.imgviz).convert('RGB')

        sample = transform_delete_gray(image)  # torch.Size([3, 224, 224])
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

        transform = T.Compose([T.Resize([224, 224]),
                               # T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = Image.open(args.imgviz).convert('RGB')
        sample = transform(image).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])

        sample_test = transform1(image).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])
        sample_test.requires_grad = False

        out_test = model(sample_test)  # torch.Size([1, 1000])

        pred_label = out_test.argmax(dim=1).item()
        if 'imagenet' in args.dataset:
            pred_class = dataset_crp.dataset_split.classes[pred_label][0]
            lable_name = dataset_crp.dataset_split.classes[labels_source[0]][0]
        else:
            raise ValueError('dataset not defined')

        file_name = f'img{img_idx}_label{labels_source[0]}_{lable_name.replace(" ", "-")}__' \
                     f'pred{pred_label}_{pred_class.replace(" ", "-")}'
        sample_dir = os.path.join(crp_base_dir, file_name)

        if os.path.exists(f'{sample_dir}') is True:
            eval_num += 1
            pass
        else:
            continue

        CoE_eval_all_dict[file_name] = {}
        CoE_eval_total_score_dict[file_name] = {}
        CoE_eval_score1_dict[file_name] = {}
        CoE_eval_score2_dict[file_name] = {}
        CoE_eval_score3_dict[file_name] = {}

        if os.path.exists(f'{sample_dir}/final_response_random.json'):
            pass
        else:
            print(sample_dir)
            if not os.path.exists(sample_dir):
                file_name_list = os.listdir(crp_base_dir)
                file_name_test = f'img{img_idx}_label{labels_source[0]}_{dataset_crp.classes[labels_source[0]][0].replace(" ", "-")}'

                for file_name_ready in file_name_list:
                    if file_name_test in file_name_ready:
                        sample_dir = os.path.join(crp_base_dir, file_name_ready)
                        break

        img_viz_path = args.imgviz

        ###############################################################################################################
        ###############################################################################################################
        # ************************************************************************************************
        # ******  Start CoE-eval  ******
        # *

        answer_single_file = os.path.join(sample_dir, f'coe_eval_answer_{args.atom_select}.json')
        if os.path.exists(answer_single_file) is True and b_re_eval is False:
            with open(answer_single_file, 'r') as f:
                answer_dict = json.load(f)

        else:
            answer_dict = {}
            for eval_method in eval_methods:
                if eval_method == 'random':
                    args.b_use_milan_dec = False
                    args.atom_select = 'random'
                elif eval_method == 'llm':
                    args.b_use_milan_dec = False
                    args.atom_select = 'llm'

                else:
                    raise NotImplementedError

                if args.atom_select == 'random':
                    final_explan_path = os.path.join(sample_dir, 'final_response_random.json')
                    coe_info_dict_path = os.path.join(sample_dir, 'coe_info_dict_random.json')
                elif args.atom_select == 'llm':
                    final_explan_path = os.path.join(sample_dir, 'final_response_llm.json')
                    coe_info_dict_path = os.path.join(sample_dir, 'coe_info_dict_llm.json')
                else:
                    raise NotImplementedError

                img_file = img_viz_path

                with open(final_explan_path, 'r') as f:
                    final_response_dict = json.load(f)
                final_prompt = final_response_dict['final_prompt']
                final_explan = final_response_dict['final_response']

                with open(coe_info_dict_path, 'r') as f:
                    coe_info_dict = json.load(f)
                Input_Label = coe_info_dict['lable_name']
                Prediction = coe_info_dict['pred_class']

                prompt_system = '' \
                    'You are now a scorer for an interpretability evaluation system assessing a deep visual model interpreter.'

                prompt_coe_eval_template = '' \
                   'You are now a scorer for an interpretability evaluation system assessing a deep visual model interpreter.' \
                   'This interpreter provides natural language explanations of decision-making process of a deep visual model when given an image input. ' \
                   'Your task is to evaluate and score the output explanation of the interpreter based on specified criteria to determine its quality. \n' \
                   'Your input information includes: \n' \
                   'A. Input image to the deep visual model is given as follows. \n' \
                   'B. The Ground Truth Label of the input image is {Input_Label}. \n' \
                   'C. The Prediction made by the deep visual model for this input image is {Prediction}. \n' \
                   'D. Explanation provided by the interpreter is {Explnation}. \n' \
                   '' \
                   'Based on the four pieces of information provided above, score Explanation D according to the following three Criterias. ' \
                   'Each Criteria has its own scoring rules, and you need to score Explanation D according to the standards of each Criteria: \n' \
                   'Criteria 1. Accuracy [0-2 points]:' \
                   '2 points: Almost all relevant explanation focused on key decision points, essential features, important regions, and backgroud information, ' \
                                           'with no extraneous or irrelevant content.' \
                   '1 point: Explanation is generally relevant but may contain some minor off-topic or unnecessary information.' \
                   '0 points: Explanation includes a significant amount of irrelevant content, diverging from the model’s decision-making process, impairing comprehension.\n' \
                   'Criteria 2. Completeness [0-2 points]:' \
                   '2 points: Comprehensive explanation covering all major steps, key features, backgroud information, and relevant concepts of the model’s decision process.' \
                   '1 point: Explanation addresses primary decision steps but may slightly overlook some information or secondary features.' \
                   '0 points: Incomplete explanation lacking essential decision steps or information, making comprehension challenging.\n' \
                   'Criteria 3. User Interpretability [0-2 points]:' \
                   '2 points: Explanation allows users without specialized knowledge to understand the model’s decision logic, with clear, straightforward language and smooth readability.' \
                   '1 point: Explanation is mostly understandable to users with a technical background; it is fairly clear but may require some re-reading due to less fluent phrasing or logic.' \
                   '0 points: Explanation is difficult to comprehend, with disorganized or unclear language that obscures the decision process of the model.\n' \
                   '' \
                   'Please first provide an evidence of your evaluation for each criteria, and then provide your score for each criteria, ' \
                   'avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.' \
                   'Then sum up the above scores of the three criterias as total score, the total score should follow these rules:\n' \
                   'Full Score (6 points): Explanation accurately describes critical features of the model’s decision process, with clear logic and complete structure. ' \
                   'The content is highly relevant and understandable.\n' \
                   'Medium Score (3-5 points): Explanation is generally accurate and clear, though some details may be slightly unclear or lacking in feature description.\n' \
                   'Low Score (0-2 points): Explanation is disorganized or incomplete, making it difficult for users to understand the actual decision-making process of the model.\n' \
                   'Finally, output 7 lines indicating the Evidence and the Scores for Criteria 1, Evidence and the Scores for Criteria 2' \
                   'Evidence and the Scores for Criteria 3, and the total score, respectively.' \
                   'Your responce should strictly follow the scoring rules and Output with the following format strictly:' \
                   'The Score_Evidence_1: [your score Evidence for Criteria 1]' \
                   'The score of Criteria 1: [score 1 only]' \
                   'The Score_Evidence_2: [your score Evidence for Criteria 2]' \
                   'The score of Criteria 2: [score 2 only]' \
                   'The Score_Evidence_3: [your score Evidence for Criteria 3]' \
                   'The score of Criteria 3: [score 3 only]' \
                   'The total score: [total score]' \
                   'Now, Please provide your response:'

                prompt_coe_eval = prompt_coe_eval_template.format(**{
                    "Prediction": Prediction,
                    "Input_Label": Input_Label,
                    "Explnation": final_explan,
                })
                answer_single_dict = coe_eval_llm_model.inference(img_file, prompt_coe_eval, system_prompt=prompt_system)
                print('GPT Done!')
                answer_dict[eval_method] = answer_single_dict
                CoE_eval_total_score_dict[file_name][eval_method] = answer_single_dict['total_score']
                CoE_eval_score1_dict[file_name][eval_method] = answer_single_dict['score_1']
                CoE_eval_score2_dict[file_name][eval_method] = answer_single_dict['score_2']
                CoE_eval_score3_dict[file_name][eval_method] = answer_single_dict['score_3']

            with open(answer_single_file, 'w') as f:
                json.dump(answer_dict, f, indent=4)
        CoE_eval_all_dict[file_name] = answer_dict

    with open(answer_all_file, 'w') as f:
        json.dump(CoE_eval_all_dict, f, indent=4)
    with open(answer_total_score_file, 'w') as f:
        json.dump(CoE_eval_total_score_dict, f, indent=4)

    with open(answer_score_1_file, 'w') as f:
        json.dump(CoE_eval_score1_dict, f, indent=4)
    with open(answer_score_2_file, 'w') as f:
        json.dump(CoE_eval_score2_dict, f, indent=4)
    with open(answer_score_3_file, 'w') as f:
        json.dump(CoE_eval_score3_dict, f, indent=4)

coe_eval_stat_dict = {}
coe_eval_stat_dict['total_score'] = {}
coe_eval_stat_dict['score_1'] = {}
coe_eval_stat_dict['score_2'] = {}
coe_eval_stat_dict['score_3'] = {}
for eval_method in eval_methods:
    coe_eval_stat_dict['total_score'][eval_method] = []
    coe_eval_stat_dict['score_1'][eval_method] = []
    coe_eval_stat_dict['score_2'][eval_method] = []
    coe_eval_stat_dict['score_3'][eval_method] = []

for i_img, img_coe_dict in CoE_eval_all_dict.items():
    for eval_method in eval_methods:
        if eval_method in img_coe_dict.keys():
            coe_eval_stat_dict['total_score'][eval_method].append(img_coe_dict[eval_method]['total_score'])
            coe_eval_stat_dict['score_1'][eval_method].append(img_coe_dict[eval_method]['score_1'])
            coe_eval_stat_dict['score_2'][eval_method].append(img_coe_dict[eval_method]['score_2'])
            coe_eval_stat_dict['score_3'][eval_method].append(img_coe_dict[eval_method]['score_3'])

coe_eval_avg_dict = {}
coe_eval_avg_dict['total_score'] = {}
coe_eval_avg_dict['score_1'] = {}
coe_eval_avg_dict['score_2'] = {}
coe_eval_avg_dict['score_3'] = {}
for eval_method in eval_methods:
    coe_eval_avg_dict['total_score'][eval_method] = np.average(coe_eval_stat_dict['total_score'][eval_method])
    coe_eval_avg_dict['score_1'][eval_method] = np.average(coe_eval_stat_dict['score_1'][eval_method])
    coe_eval_avg_dict['score_2'][eval_method] = np.average(coe_eval_stat_dict['score_2'][eval_method])
    coe_eval_avg_dict['score_3'][eval_method] = np.average(coe_eval_stat_dict['score_3'][eval_method])

print('Done!')

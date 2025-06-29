# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2022- 09- 22
Authors: Yu wenlong  and  DRAGON_501

*************************************Import***********************************"""
import argparse


def get_args_parser(stages=['CoE']):
    # for visualization or evaluation
    parser = argparse.ArgumentParser(description='Chain-of-Explanation, CoE', add_help=False)
    str2bool = lambda x: x.lower() == "true"

    parser.add_argument('--other', '-ot', default='ckptdownload')

    parser.add_argument('--out_forder_name', '-ofn', default='pretrained')
    parser.add_argument('--out_forder_type', '-oft', default='o')

    parser.add_argument('--model_name', '-model', default='rn152')
    parser.add_argument('--resume', help='resume from checkpoint',
                        # default=None,
                        default='rn152',
                        )
    parser.add_argument('--total_process_steps', default=1)
    parser.add_argument('--k', type=int, default=52)

    parser.add_argument('--b_comp_LRP_CRP', type=str2bool, default=True)
    parser.add_argument('--b_store_LRP_CRP', type=str2bool, default=True)
    parser.add_argument('--b_force_store_LRP_CRP', type=str2bool, default=False)

    parser.add_argument('--bn_pltshow', type=str2bool, default=True)

    parser.add_argument('--main_ckpt_root', '-mres', default='')

    parser.add_argument('--topki', default=4)
    parser.add_argument('--b_norm', type=str2bool, default=True)
    parser.add_argument('--batchsize', default=4)

    parser.add_argument('--dataset', default='imagenet-val', choices=['imagenet-val'])
    parser.add_argument('--data_path', default='./dataset')
    parser.add_argument('--ckpt_dir', default='')

    parser.add_argument('--alleval_dir', default='')
    parser.add_argument("--gpu_location", '-gpul', default='other')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

    parser.add_argument('--device', type=str, default="cuda")

    # coe model parser

    parser.add_argument('--mllm_name', type=str, default="intern", choices=["intern"])

    parser.add_argument('--b_sem_entropy', type=str2bool, default=True)

    parser.add_argument('--b_compute_polysem_desc', type=str2bool, default=False)
    parser.add_argument('--b_compute_polysem_entropy', type=str2bool, default=False)
    #

    parser.add_argument('--b_compute_polysem_desc_lvlm', type=str2bool, default=True)
    parser.add_argument('--b_force_compute_polysem_desc_lvlm', type=str2bool, default=False)

    parser.add_argument('--b_store_ccid_sem_imgs', type=str2bool, default=False)

    parser.add_argument('--polysem_model_vlm', type=str, default="gpt-4o-2024-08-06",
                        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4o-2024-08-06',
                                 'intern'])

    # entailment parser
    parser.add_argument("--b_sem_entropy_entail", type=str2bool, default=False)
    parser.add_argument("--b_force_sem_entropy_entail", type=str2bool, default=False)
    parser.add_argument("--entail_model", default='deberta', type=str, choices=['deberta'])
    parser.add_argument('--strict_entail', type=str2bool, default=True)

    parser.add_argument('--cctype', type=str, default="relevance", choices=['relevance', 'activation'])

    parser.add_argument('--force_layer', type=str, default=None)
    parser.add_argument('--target_num', type=int, default=0)

    parser.add_argument('--b_milan', type=str2bool, default=False)
    parser.add_argument('--b_use_milan_dec', type=str2bool, default=False)
    parser.add_argument('--b_othermethod', type=str, choices=['milan'])

    parser.add_argument('--atom_select', type=str, default="llm", choices=['llm', 'random'])

    parser.add_argument('--atom_select_llm', type=str, default="gpt-4-turbo",
                        choices=['gpt-4-turbo', 'gpt-3.5-turbo-1106', 'intern'])

    parser.add_argument('--coe_final_llm', type=str, default="gpt-4-turbo", choices=['gpt-4-turbo'])

    # Evaluation parser
    parser.add_argument('--cpe_eval_vlm', type=str, default="gpt-4o-2024-08-06",
                        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4o-2024-08-06', 'intern'])

    parser.add_argument('--coe_eval_llm', type=str, default="gpt-4o-2024-08-06",
                        choices=['gpt-4-turbo', 'gpt-3.5-turbo-1106','gpt-4o-2024-08-06', 'intern'])

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------

    return parser





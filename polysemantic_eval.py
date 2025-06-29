# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2024- 09- 19
Authors: Yu wenlong  and  DRAGON_501
Link:

*************************************Import***********************************"""

import json

import logging
import os

import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd

from semantic_entropy import get_semantic_ids, EntailmentDeberta

from closeai import OpenAIGpt
from util.plot_utils import clear_plot

"""**********************************Import***********************************"""
'''***************************************************************************'''


def plt_entropy(probs, names, title=None,
                xlabel='Semantic Name', ylabel='Probability of Semantic',
                text=[f'Naive Semantic Entropy'],
                b_plt_show=True, save_path=None):
    x = range(len(probs))

    ax = plt.gca()
    plt.plot(x, probs, marker='o')
    if title is not None:
        plt.title(title)
    plt.xticks(x, names, rotation=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, i_txt in enumerate(text):
        ax.text(0.07, i*0.1+0.6, i_txt, fontsize=12, ha='left', transform=ax.transAxes)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, )
    if b_plt_show is True:
        plt.show()
    plt.close()


class Describe_Polysemantic():

    def __init__(self, model_name=None, args=None, b_format=True, api_key=None, base_url=None, temperature=0.1):
        if 'gpt' in model_name:
            self.Polysem_model = OpenAIGpt(model_name=model_name, b_format=True, temperature=temperature)

    def infer_polysem(self, images, prompt, b_show=False, img_store_path=''):
        answer = self.Polysem_model.inference(images=images, prompt=prompt, b_show=b_show, img_store_path=img_store_path)

        if isinstance(answer, str):
            answer = answer.lstrip()
        elif isinstance(answer, dict):
            pass

        return answer


class Explain_LVLM():

    def __init__(self, model_name=None, args=None, b_format=True, api_key=None, base_url=None, temperature=0.1):
        if 'gpt' in model_name:
            self.LVL_Model = OpenAIGpt(model_name=model_name, b_format=b_format, temperature=temperature)

    def infer_lvl(self, images, prompt, system_prompt=None, resolution='low', b_show=False):
        answer = self.LVL_Model.inference(images=images, prompt=prompt, system_prompt=system_prompt,
                                          resolution='low', b_show=b_show)

        if isinstance(answer, str):
            answer = answer.lstrip()
        elif isinstance(answer, dict):
            pass

        return answer


def check_folder(path):
    if os.path.isdir(path):
        file_names = os.listdir(path)
        all_jpeg = True
        return_dict_all = {}
        return_dict_img = {}
        return_dict_not_img = {}
        for file_name in file_names:
            file_path = os.path.join(path, file_name)

            if os.path.isfile(file_path):
                if file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.jpg'):
                    all_jpeg = True
                    return_dict_img[file_name] = file_path
                else:
                    all_jpeg = False
                    return_dict_not_img[file_name] = file_path
                return_dict_all[file_name] = file_path

            elif os.path.isdir(file_path):

                sub_dict_all, sub_dict_img, sub_dict_not_img = check_folder(file_path)
                return_dict_all[file_name] = sub_dict_all
                return_dict_img[file_name] = sub_dict_img
                return_dict_not_img[file_name] = sub_dict_not_img
            else:
                return_dict_all, return_dict_img, return_dict_not_img = None, None, None

        return return_dict_all, return_dict_img, return_dict_not_img
    return None, None, None


def main_polysemantic(args, img_store_paths=None, ccid_desc_path=None, b_show=False,
                      target_ccids=None, show_save_path=None):

    # Note: This function has the function of resuming according to the storation in the accordingly folder.
    #

    vlm_model_name = args.polysem_model_vlm

    if args.other == 'debug' or b_show is True or target_ccids is not None:
        b_show = True
    else:
        b_show = False

    if img_store_paths is None:
        img_store_paths = './tmp_api_imgs'
        image_files = []
        for i in range(1, 8):
            image_files.append(f'./tmp_api_imgs/image{4}.jpg')
        for i in range(8, 16):
            image_files.append(f'./tmp_api_imgs/image{9}.jpg')
        img_store_paths_dict = {'eval_imgs': image_files}
        img_15_store_paths = img_store_paths

    else:
        img_store_paths_dict, _, _ = check_folder(img_store_paths)
        img_15_store_paths = os.path.dirname(img_store_paths)

    # ##
    # ***********************************************************************************************
    # *********************************************************************

    #  Check the files in the folder
    img_ly_ccid_dict = {}
    for ly_name, ly_paths in img_store_paths_dict.items():

        if args.force_layer is not None:
            if ly_name != args.force_layer:
                continue

        img_ly_ccid_dict[ly_name] = {}
        if len(ly_paths) > 15:
            assert isinstance(ly_paths, list) or isinstance(ly_paths, dict), f'ly_paths should be list or dict, but got {type(ly_paths)}'

            for i_img, (img_name, img_path) in enumerate(ly_paths.items()):
                if not os.path.isfile(img_path):
                    logging.warning(f'File {ly_paths[i_img]} not found.')
                    ly_paths.pop(i_img)
                    continue

                ccid_i_ccid = img_name.split('.')[0]
                ccid, i_ccid = ccid_i_ccid.split('_')
                if ccid not in img_ly_ccid_dict[ly_name].keys():
                    img_ly_ccid_dict[ly_name][ccid] = {int(i_ccid): img_path}
                else:
                    img_ly_ccid_dict[ly_name][ccid][int(i_ccid)] = img_path

        elif len(ly_paths) <= 15:
            for i_img, img_path in enumerate(ly_paths):
                if not os.path.isfile(img_path):
                    logging.warning(f'File {ly_paths[i_img]} not found.')
                    ly_paths.pop(i_img)
                    continue

                if 'x_ccid' not in img_ly_ccid_dict[ly_name].keys():
                    img_ly_ccid_dict[ly_name]['x_ccid'] = [img_path]
                else:
                    img_ly_ccid_dict[ly_name]['x_ccid'].append(img_path)

        elif len(ly_paths) == 0:
            logging.warning(f'No image files found in {ly_name} folder.')

    # ***********************************************************************************************
    # Definition of Entailment Model
    #

    args.b_sem_entropy_entail = True
    if args.b_force_sem_entropy_entail is True:
        args.b_sem_entropy_entail = True
    if args.b_sem_entropy_entail is True:
        logging.info('Begin loading entailment model.')
        if args.entail_model == 'deberta':
            entail_model = EntailmentDeberta()
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    target_ccids_dict = {}
    if target_ccids is not None:
        for (ly_name, ccid_name) in target_ccids:
            if ly_name not in target_ccids_dict.keys():
                target_ccids_dict[ly_name] = {}
            target_ccids_dict[ly_name][ccid_name] = {}

    print('Ready~ Go!')

    # ################################################################################################
    # ***********************************************************************************************
    # *********************************************************************
    # 1. Decouple and Obtain polysemantical interpretations of each single neuron (VC)
    #
    # Note: All of these have the function of resuming from where they were interrupted.
    # You can Directly read the pre-computed polysemantic description file when all the following swithes are False.
    #
    #

    # args.b_compute_polysem_desc = True  # If True, the polysemantical description of each VC will be computed.
    # args.b_compute_polysem_desc_lvlm = True

    if args.b_compute_polysem_desc is True or args.b_force_compute_polysem_desc_lvlm is True:
        if args.b_force_compute_polysem_desc_lvlm is True:
            args.b_compute_polysem_desc = True
        i_cccc = 0
        sem_desc = {}

        # Define the polysem concept atoms description model
        Polysem_model = Describe_Polysemantic(model_name=vlm_model_name, args=args, b_format=True, temperature=0.1)

        for ly_name, ly_paths in img_ly_ccid_dict.items():

            if args.force_layer is not None:
                if ly_name != args.force_layer:
                    continue

            i_cccc = 0
            if not os.path.exists(os.path.join(ccid_desc_path, vlm_model_name, ly_name)) and \
                    args.b_compute_polysem_desc_lvlm is False:
                continue
            if ly_name not in sem_desc.keys():
                sem_desc[ly_name] = {}

            img_15_store_path_lyname = os.path.join(img_15_store_paths, 'img_15', ly_name)
            os.makedirs(img_15_store_path_lyname, exist_ok=True)

            # for better parallelism, we only process 256 images per layer at a time.
            # You can call this function multiple times parallelly with different args.target_num.
            # For example, # CUDA_VISIBLE_DEVICES=0 python main.py --force_layer layer4 --target_num 0
            target_num = args.target_num
            target_num = target_num * 256

            for ccid, image_files in tqdm(ly_paths.items(), desc=f"Polysem_desc_{ly_name}"):

                if args.other == 'debug':
                    if i_cccc > 2:
                        break

                i_cccc += 1
                if args.force_layer is not None:
                    if target_num <= i_cccc < target_num + 256:
                        pass
                    else:
                        continue

                #
                # ***********************************************************************************************
                # ***********************************************************************************************
                # 1.1
                # Store 15 images onto one picture and save it. Only for better visualization.

                img_15_store_path = os.path.join(img_15_store_path_lyname, f'{ccid}.jpg')

                if os.path.exists(img_15_store_path) is False:

                    if b_show is True or img_15_store_path != '':
                        fig, axs = plt.subplots(3, 5, figsize=(15, 9), dpi=200)

                        for i, ax in enumerate(axs.flat):
                            if i < len(image_files):
                                img = mpimg.imread(image_files[i])
                                ax.imshow(img)
                                ax.axis('off')
                            else:
                                ax.axis('off')

                        plt.tight_layout()
                        if img_15_store_path != '':
                            plt.savefig(img_15_store_path, bbox_inches='tight', pad_inches=0)

                        clear_plot()

                #
                # ***********************************************************************************************
                # ***********************************************************************************************
                # *
                # * 1.2 Call GPT-4o to generate a description for each VC.
                # If the JSON file already exists and re-computation is not required,
                # then it is sufficient to read the JSON file.
                #
                if os.path.exists(os.path.join(ccid_desc_path, vlm_model_name, ly_name, f'description_{ccid}.json'))\
                        and args.b_force_compute_polysem_desc_lvlm is False:

                    if ccid_desc_path is not None:
                        file_name = os.path.join(ccid_desc_path, vlm_model_name, ly_name, f'description_{ccid}.json')
                    else:
                        file_name = f'{img_store_paths}/answer_{vlm_model_name}_{ccid}.json'
                    with open(file_name) as file_obj:
                        se_1cc_dict = json.load(file_obj)
                        sem_desc[ly_name][ccid] = se_1cc_dict

                else:  # If the JSON file does not exist or if a forced recalculation is required,
                    # the method will be called to perform the recalculation.
                    # ***********************************************************************************************
                    # *********************************************************************

                    if args.b_compute_polysem_desc_lvlm is True or args.b_force_compute_polysem_desc_lvlm is True:

                        #
                        prompt_vlm_poly_individual = \
                            'Given 15 images, each containing highlighted regions, find some common objects and attributes in these images ' \
                            'and describe each image with words especially repeated across these images.\n' \
                            'Your response should follow these rules: ' \
                            '1. Pay more attention to the repeated objects or attributes across these images. ' \
                            '2. Possible objects or attributes you can use to describe these images are ' \
                            'object category, scene, object part, colour, texture, material, position,' \
                            'transparency, brightness, shape, size, edges, and their relationships. ' \
                            '3. The identified common objects or attributes must appear simultaneously in at least 5 images. ' \
                            '4. The identified specific objects or attributes represent some important contents of an individual image ' \
                            'but not in the common objects or attributes found in the previous step. ' \
                            '5. Your description of each image should be simple and only 3 words. ' \
                            '6. Your response should be in the format of a JSON file, of which each key is a simple image index and ' \
                            'each value is an object or attribute.\n' \
                            'Your identification process should strictly follow these steps: ' \
                            'Step 1, take an overview of all 15 images and summarize all possible common objects or attributes that appear simultaneously in at least any 5 of these images. ' \
                            'Step 2, for each individual image, identify the common objects or attributes found in Step 1 that are also appear in the current image to describe the current image.' \
                            'Step 3, for each individual image, you can also use some specific attributes or objects that are not common across these images to describe the current image ' \
                            'if there is not enough 3-word description for the common object or attribute found in Step 2.\n' \
                            'Now, please provide your response: '

                        answer = Polysem_model.infer_polysem(images=image_files, prompt=prompt_vlm_poly_individual,
                                                             b_show=b_show, img_store_path=img_15_store_path)

                        if ccid_desc_path is not None:
                            os.makedirs(os.path.join(ccid_desc_path, vlm_model_name, ly_name), exist_ok=True)
                            file_name = os.path.join(ccid_desc_path, vlm_model_name, ly_name, f'description_{ccid}.json')
                        else:
                            file_name = f'{img_store_paths}/answer_{vlm_model_name}_{ccid}.json'
                        with open(file_name, 'w') as file_obj:
                            json.dump(answer, file_obj)

                        se_1cc_dict = answer
                        sem_desc[ly_name][ccid] = se_1cc_dict

        sem_desc_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'polysemantic_desc_all_dict.json')
        os.makedirs(os.path.join(ccid_desc_path, vlm_model_name), exist_ok=True)
        with open(sem_desc_file_name, 'w') as file_obj:
            json.dump(sem_desc, file_obj)

    # Or Directly read the pre-computed polysemantic description file.
    else:
        sem_desc_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'polysemantic_desc_all_dict.json')
        if not os.path.exists(sem_desc_file_name):
            raise ValueError(f'No polysemantic description file found in {sem_desc_file_name}.')
        with open(sem_desc_file_name) as file_obj:
            sem_desc = json.load(file_obj)

    #
    # ***********************************************************************************************
    # ***********************************************************************************************
    # ***********************************************************************************************

    # Obtain the description of the target CCID and store it in a dictionary, which will facilitate subsequent display.
    for ly_name, ccid_dict in target_ccids_dict.items():
        for ccid_name, ccid_dict_2 in ccid_dict.items():
            target_ccids_dict[ly_name][int(ccid_name)] = sem_desc[ly_name][str(ccid_name)]

    #
    # ################################################################################################
    # ################################################################################################
    # ***********************************************************************************************
    # *********************************************************************
    # 2. Calculate the Concept Atom Probability and the Concept polysemanticity entropy (CPE)
    #
    #  Note: All of these have the function of resuming from where they were interrupted.
    #  You can Directly read the pre-computed polysemantic description file when all the following swithes are False.
    #

    # args.b_compute_polysem_entropy = True
    sem_ccids_atoms_probs_dict = {}
    if args.b_compute_polysem_entropy is True:
        semantic_entropys = {}
        semantic_id_path = os.path.join(ccid_desc_path, vlm_model_name, f'entail_{args.entail_model}')
        os.makedirs(semantic_id_path, exist_ok=True)

        for ly_name, ccid_dict in sem_desc.items():
            if ly_name not in semantic_entropys.keys():
                semantic_entropys[ly_name] = {}
            if ly_name not in sem_ccids_atoms_probs_dict.keys():
                sem_ccids_atoms_probs_dict[ly_name] = {}

            if args.force_layer is not None:
                if ly_name != args.force_layer:
                    continue

            semantic_id_lyname_path = os.path.join(semantic_id_path, ly_name)
            os.makedirs(semantic_id_lyname_path, exist_ok=True)

            target_num = args.target_num
            target_num = target_num * 256
            j_cccc = 0
            for ccid, se_1cc_dict in tqdm(ccid_dict.items(), desc=f"Polysem_Entropy_{ly_name}"):

                j_cccc += 1
                if args.force_layer is not None:
                    if target_num <= j_cccc < target_num + 256:
                        pass
                    else:
                        continue

                if ccid not in semantic_entropys[ly_name].keys():
                    semantic_entropys[ly_name][ccid] = {}
                if ccid not in sem_ccids_atoms_probs_dict[ly_name].keys():
                    sem_ccids_atoms_probs_dict[ly_name][ccid] = {}

                # *********************************************************************
                # 2.1 directly calculating the overlap based on the string form, removing the overlap,
                # achieved by using the set collection to eliminate the overlap.
                #

                se_1cc_set = set()
                se_1cc_list = list()
                for key, value in se_1cc_dict.items():
                    if isinstance(value, str):
                        se_1cc_list.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            se_1cc_list.append(item)
                se_1cc_list = sorted(se_1cc_list)
                se_1cc_set = set(se_1cc_list)

                #
                # ***********************************************************************************************
                # ***********************************************************************************************
                # *********************************************************************
                # 2.3 Calculate the semantic probability distribution
                # 2.3.1 Naive version

                sem_len = len(se_1cc_list)
                sem_counts_dict = Counter(se_1cc_list)
                sem_probs_naive_dict = {element: count / sem_len for element, count in sem_counts_dict.items()}
                sem_probs_naive_dict = dict(sorted(sem_probs_naive_dict.items(), key=lambda item: item[1]))
                for key in sem_probs_naive_dict:
                    sem_probs_naive_dict[key] = round(sem_probs_naive_dict[key], 4)

                sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_naive_dict'] = sem_probs_naive_dict

                #
                # ***********************************************************************************************
                # ***********************************************************************************************
                # *********************************************************************
                # 2.2 Calculate the entailment relationship (id cluster)
                #

                se_1cc_set_list = list(se_1cc_set)
                se_1cc_set_list = sorted(se_1cc_set_list)

                semantic_id_lyname_ccid_path = os.path.join(semantic_id_lyname_path, f'semantic_ids_{ccid}.json')
                if os.path.exists(semantic_id_lyname_ccid_path) and args.b_force_sem_entropy_entail is False:
                    with open(semantic_id_lyname_ccid_path) as file_obj:
                        semantic_ids = json.load(file_obj)
                else:
                    semantic_ids = get_semantic_ids(
                        se_1cc_set_list, model=entail_model, strict_entailment=args.strict_entail, example=None)
                    with open(semantic_id_lyname_ccid_path, 'w') as file_obj:
                        json.dump(semantic_ids, file_obj)

                if len(semantic_ids) != len(se_1cc_set_list):
                    raise ValueError(f'The number of semantic ids is not equal to the number of semantic elements.')

                #
                # ***********************************************************************************************
                # *********************************************************************
                # 2.3 Calculate the semantic probability distribution after clustering,
                # and the semantic dictionary becomes the first semantic words.
                #

                se_1cc_set_list_cluster = se_1cc_set_list.copy()
                sem_counts_cluster_dict = sem_counts_dict.copy()
                repeated_sem_name = set()
                semantic_ids_list = list(semantic_ids)

                already_precessed = []

                for i_id, iid in enumerate(semantic_ids_list):
                    if iid in already_precessed:
                        continue
                    i_sem_name = se_1cc_set_list[i_id]

                    for j_id in range(i_id+1, len(semantic_ids_list)):

                        if iid == semantic_ids[j_id]:
                            already_precessed.append(semantic_ids[j_id])
                            j_sem_name = se_1cc_set_list_cluster[j_id]

                            prob_i = sem_probs_naive_dict[i_sem_name]
                            prob_j = sem_probs_naive_dict[j_sem_name]

                            sem_counts_cluster_dict[j_sem_name] = 0
                            sem_counts_cluster_dict[i_sem_name] += sem_counts_dict[j_sem_name]
                            se_1cc_set_list_cluster[j_id] = i_sem_name
                            repeated_sem_name.add(j_sem_name)

                sem_len_cluster = np.sum(list(sem_counts_cluster_dict.values()))
                se_1cc_set_list_cluster = set(se_1cc_set_list_cluster)
                for name in repeated_sem_name:
                    sem_counts_cluster_dict.pop(name)

                sem_probs_cluster_dict = {element: count / sem_len for element, count in sem_counts_cluster_dict.items()}

                sum_sem_probs_cluster = np.sum(list(sem_probs_cluster_dict.values()))
                sum_sem_nums_cluster = np.sum(list(sem_counts_cluster_dict.values()))
                assert (np.abs(sum_sem_probs_cluster - 1) < 1e-3), 'Probabilities do not sum to 1.'
                assert (sum_sem_nums_cluster == sem_len), 'Counts do not sum to sem_len.'

                sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_cluster_dict'] = sem_probs_cluster_dict

                #
                # ***********************************************************************************************
                # ***********************************************************************************************
                # *********************************************************************
                # 2.4 Calcupate the CPE
                # 2.4.1 Naive version

                sem_probs_naive_dict_sorted = dict(sorted(sem_probs_naive_dict.items(), key=lambda item: item[1]))

                sem_probs = list(sem_probs_naive_dict_sorted.values())
                sem_names = list(sem_probs_naive_dict_sorted.keys())
                # sum_sem_probs = np.sum(sem_probs)
                assert (np.abs(np.sum(sem_probs)) - 1) < 1e-3, f'Probabilities do not sum to 1. ccid: {ly_name} {ccid}'

                sem_entropy_naive = (-np.sum(sem_probs * np.log(sem_probs)))
                sem_entropy_naive_log = (-np.sum(sem_probs * np.log(sem_probs))) / np.log(len(sem_probs))

                semantic_entropys[ly_name][ccid]['sem_entropy_naive'] = sem_entropy_naive
                semantic_entropys[ly_name][ccid]['sem_entropy_naive_log'] = sem_entropy_naive_log

                #
                #
                # *********************************************************************
                # 2.4.2 Calculate the completed naive polysemy entropy

                num_sem_min = 15
                len_sem_naive = len(sem_counts_dict.keys())
                sem_counts_dict_pad = sem_counts_dict.copy()
                if len_sem_naive < num_sem_min:
                    for i_added in range(len_sem_naive, num_sem_min):
                        sem_counts_dict_pad[f'new_sem_{i_added+1}'] = 1

                sem_pad_len = sum(list(sem_counts_dict_pad.values()))

                sem_probs_naive_pad_dict = {element: count / sem_pad_len for element, count in sem_counts_dict_pad.items()}

                sem_probs_naive_pad_dict_sorted = dict(sorted(sem_probs_naive_pad_dict.items(), key=lambda item: item[1]))

                sem_naive_pad_probs = list(sem_probs_naive_pad_dict_sorted.values())
                sem_naive_pad_names = list(sem_probs_naive_pad_dict_sorted.keys())
                # sum_sem_probs = np.sum(sem_probs)
                assert (np.abs(np.sum(sem_naive_pad_probs)) - 1) < 1e-3, 'Probabilities do not sum to 1.'

                sem_entropy_naive_pad = (-np.sum(sem_naive_pad_probs * np.log(sem_naive_pad_probs)))
                sem_entropy_naive_pad_log = (-np.sum(sem_naive_pad_probs * np.log(sem_naive_pad_probs))) / np.log(len(sem_naive_pad_probs))

                semantic_entropys[ly_name][ccid]['sem_entropy_naive_pad'] = sem_entropy_naive_pad
                semantic_entropys[ly_name][ccid]['sem_entropy_naive_pad_log'] = sem_entropy_naive_pad_log

                sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_naive_pad_dict'] = sem_probs_naive_pad_dict

                #
                # ***********************************************************************************************
                # *********************************************************************
                # 2.4.3 Calculate the post-clustering CPE
                #

                sem_probs_cluster_dict_sorted = dict(sorted(sem_probs_cluster_dict.items(), key=lambda item: item[1]))

                sem_clusters_probs = list(sem_probs_cluster_dict_sorted.values())
                sem_clusters_names = list(sem_probs_cluster_dict_sorted.keys())

                sem_entropy_clusters = (-np.sum(sem_clusters_probs * np.log(sem_clusters_probs)))
                sem_entropy_clusters_log = (-np.sum(sem_clusters_probs * np.log(sem_clusters_probs))) / (np.log(len(sem_clusters_probs))+1e-3)
                if sem_entropy_clusters_log == np.nan:
                    sem_entropy_clusters_log = 0.0

                semantic_entropys[ly_name][ccid]['sem_entropy_clusters'] = sem_entropy_clusters
                semantic_entropys[ly_name][ccid]['sem_entropy_clusters_log'] = sem_entropy_clusters_log

                # *********************************************************************
                # 2.4.4 Calculate the completed clustering CPE
                #

                num_sem_min = 15
                len_sem_cluster = len(sem_counts_cluster_dict.keys())
                sem_counts_cluster_dict_pad = sem_counts_cluster_dict.copy()
                if len_sem_cluster < num_sem_min:
                    for i_added in range(len_sem_cluster, num_sem_min):
                        sem_counts_cluster_dict_pad[f'new_sem_{i_added+1}'] = 1

                sem_cluster_pad_len = sum(list(sem_counts_cluster_dict_pad.values()))

                sem_probs_cluster_pad_dict = {element: count / sem_cluster_pad_len for element, count in sem_counts_cluster_dict_pad.items()}

                sem_probs_cluster_pad_dict_sorted = dict(sorted(sem_probs_cluster_pad_dict.items(), key=lambda item: item[1]))

                sem_cluster_pad_probs = list(sem_probs_cluster_pad_dict_sorted.values())
                sem_cluster_pad_names = list(sem_probs_cluster_pad_dict_sorted.keys())
                # sum_sem_probs = np.sum(sem_probs)
                assert (np.abs(np.sum(sem_cluster_pad_probs)) - 1) < 1e-3, 'Probabilities do not sum to 1.'

                sem_entropy_clusters_pad = (-np.sum(sem_cluster_pad_probs * np.log(sem_cluster_pad_probs)))
                sem_entropy_clusters_pad_log = (-np.sum(sem_cluster_pad_probs * np.log(sem_cluster_pad_probs))) / np.log(len(sem_cluster_pad_probs))

                semantic_entropys[ly_name][ccid]['sem_entropy_clusters_pad'] = sem_entropy_clusters_pad
                semantic_entropys[ly_name][ccid]['sem_entropy_clusters_pad_log'] = sem_entropy_clusters_pad_log

                sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_cluster_pad_dict'] = sem_probs_cluster_pad_dict

                #
                # *********************************************************************
                # ***********************************************************************************************

        # Save the result to a file
        sem_ccid_atom_prob_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'sem_ccids_atoms_probs_dict.json')
        os.makedirs(os.path.join(ccid_desc_path, vlm_model_name), exist_ok=True)
        for ly_name, ccid_dict in sem_ccids_atoms_probs_dict.items():
            for ccid, atom_prob_dict in ccid_dict.items():
                for key, value in atom_prob_dict.items():
                    value_copy = value.copy()
                    for k, v in value.items():
                        if "\'" in k:
                            k_copy = k.replace("\'", "_")
                            value_copy[k.replace("\'", "_")] = v
                        if "\"" in k:
                            k_copy = k.replace("\"", "_")
                            value_copy[k.replace("\"", "_")] = v

                    sem_ccids_atoms_probs_dict[ly_name][ccid][key] = value_copy
        with open(sem_ccid_atom_prob_file_name, 'w') as file_obj:
            json.dump(sem_ccids_atoms_probs_dict, file_obj)

        sem_entropy_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'polysemantic_entropy_all_dict.json')
        os.makedirs(os.path.join(ccid_desc_path, vlm_model_name), exist_ok=True)
        with open(sem_entropy_file_name, 'w') as file_obj:
            json.dump(semantic_entropys, file_obj)

    else:
        # for resuming and recovering the previous results.
        # Store the multi-meaning descriptions of atoms and probabilities in a dictionary.
        # The key is the atom and the value is the probability.
        sem_ccid_atom_prob_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'sem_ccids_atoms_probs_dict.json')
        with open(sem_ccid_atom_prob_file_name) as file_obj:
            sem_ccids_atoms_probs_dict = json.load(file_obj)

        sem_entropy_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'polysemantic_entropy_all_dict.json')
        with open(sem_entropy_file_name) as file_obj:
            semantic_entropys = json.load(file_obj)

    # ################################################################################################
    # ################################################################################################

    # ################################################################################################
    # ***********************************************************************************************
    # *********************************************************************
    #
    # 3. Calculate the statistical information of entropy,
    # including the mean value of each layer and the mean value of the entire model, etc.
    #

    static_entropy_all_dict = {}
    static_entropy_avg_dict = {}
    static_entropy_avg_dict['whole_model'] = {}

    #
    if args.b_compute_polysem_entropy is True:

        for ly_name, ccid_dict in semantic_entropys.items():
            static_entropy_all_dict[ly_name] = {}
            static_entropy_avg_dict[ly_name] = {}

            for ccid, sem_entropy_ccid in ccid_dict.items():

                for key, value in sem_entropy_ccid.items():
                    if key not in static_entropy_all_dict[ly_name].keys():
                        static_entropy_all_dict[ly_name][key] = {}
                    if ccid not in static_entropy_all_dict[ly_name][key].keys():
                        static_entropy_all_dict[ly_name][key][ccid] = []

                    static_entropy_all_dict[ly_name][key][ccid] = value
                    if key not in static_entropy_avg_dict[ly_name].keys():
                        static_entropy_avg_dict[ly_name][key] = 0

            for key in static_entropy_avg_dict[ly_name].keys():
                static_entropy_avg_dict[ly_name][key] = np.mean(list(static_entropy_all_dict[ly_name][key].values()))

        for ly_name, layer_dict in static_entropy_avg_dict.items():
            if 'whole_model' in ly_name:
                continue
            for key, value in layer_dict.items():
                if key not in static_entropy_avg_dict['whole_model'].keys():
                    static_entropy_avg_dict['whole_model'][key] = []
                static_entropy_avg_dict['whole_model'][key].append(value)

        for key in static_entropy_avg_dict['whole_model'].keys():
            static_entropy_avg_dict['whole_model'][key] = np.mean(static_entropy_avg_dict['whole_model'][key])

        static_sem_entropy_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'static_polysemantic_entropy_all_dict.json')
        static_sem_entropy_avg_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'static_polysemantic_entropy_avg_dict.json')
        os.makedirs(os.path.join(ccid_desc_path, vlm_model_name), exist_ok=True)
        with open(static_sem_entropy_file_name, 'w') as file_obj:
            json.dump(static_entropy_all_dict, file_obj)
        with open(static_sem_entropy_avg_file_name, 'w') as file_obj:
            json.dump(static_entropy_avg_dict, file_obj)
    else:
        static_sem_entropy_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'static_polysemantic_entropy_all_dict.json')
        static_sem_entropy_avg_file_name = os.path.join(ccid_desc_path, vlm_model_name, 'static_polysemantic_entropy_avg_dict.json')
        with open(static_sem_entropy_file_name) as file_obj:
            static_entropy_all_dict = json.load(file_obj)
        with open(static_sem_entropy_avg_file_name) as file_obj:
            static_entropy_avg_dict = json.load(file_obj)

    #
    # ################################################################################################

    # ################################################################################################
    # ***********************************************************************************************
    # *********************************************************************
    #
    # Draw 15 pictures with the specified layer and CCID.
    #

    key_name = 'sem_probs_cluster_dict'
    if b_show is True:

        for ly_name, ly_paths in img_ly_ccid_dict.items():
            if ly_name not in target_ccids_dict.keys():
                continue
            i_cccc = 0
            for ccid in list(target_ccids_dict[ly_name].keys()):
            # for ccid, image_files in ly_paths.items():
                if str(ccid) not in list(ly_paths.keys()):
                    continue
                if args.other == 'debug':
                    if i_cccc > 2:
                        break
                i_cccc += 1

                image_files = ly_paths[str(ccid)]
                xxx = sem_ccids_atoms_probs_dict[ly_name][str(ccid)][key_name].copy()
                xxx_sorted = dict(sorted(xxx.items(), key=lambda item: item[1], reverse=True))

                print(list(xxx_sorted.keys()))
                print(list(xxx_sorted.values()))

                fig, axs = plt.subplots(3, 5, figsize=(15, 9), dpi=200)
                # fig, axs = plt.subplots(5, 3, figsize=(9, 15), dpi=200)

                plt.subplots_adjust(hspace=0.02)
                plt.subplots_adjust(wspace=0.01)

                for i, ax in enumerate(axs.flat):
                    print(i)
                    if i < len(image_files):
                        img = mpimg.imread(image_files[i])
                        ax.imshow(img)
                        ax.axis('off')
                    else:
                        ax.axis('off')

                plt.tight_layout()
                if show_save_path is not None:
                    show_save_path_ccid = os.path.join(show_save_path, f'{ly_name}-{ccid}')
                    os.makedirs(show_save_path_ccid, exist_ok=True)

                    img_15_store_path_lyname_ccid = os.path.join(show_save_path_ccid, f'{ccid}.jpg')

                    plt.savefig(img_15_store_path_lyname_ccid, bbox_inches='tight', pad_inches=0)

                clear_plot()
    # ################################################################################################
    # ***********************************************************************************************
    # *********************************************************************
    #
    # Draw the distribution chart of the several types of entropy for the given layer with the corresponding CCID
    #

    if b_show is True:
        for ly_name, ccid_dict in sem_desc.items():
            for ccid, atoms_dict in ccid_dict.items():
                if target_ccids is not None:
                    if ly_name not in list(target_ccids_dict.keys()) or int(ccid) not in list(target_ccids_dict[ly_name].keys()):
                        continue

                    if show_save_path is not None:
                        show_save_path_ccid = os.path.join(show_save_path, f'{ly_name}-{ccid}')
                        os.makedirs(show_save_path_ccid, exist_ok=True)

                        img_15_store_path_lyname = os.path.join(img_15_store_paths, 'img_15', ly_name)

                        atom_set_all = sem_desc[ly_name][str(ccid)]
                        with open(os.path.join(show_save_path_ccid, 'atoms_dict.json'), 'w') as file_obj:
                            json.dump(atom_set_all, file_obj)

                    # naive
                    sem_probs_naive_dict = sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_naive_dict']
                    sem_probs_naive_dict_sorted = dict(sorted(sem_probs_naive_dict.items(), key=lambda item: item[1]))
                    sem_probs = list(sem_probs_naive_dict_sorted.values())
                    sem_names = list(sem_probs_naive_dict_sorted.keys())
                    sem_entropy_naive = semantic_entropys[ly_name][ccid]['sem_entropy_naive']
                    sem_entropy_naive_log = semantic_entropys[ly_name][ccid]['sem_entropy_naive_log']
                    plt_entropy(probs=sem_probs, names=sem_names,
                                # title='Probability of Semantics',
                                xlabel='Semantic Name', ylabel='Probability of Semantic',
                                text=[f'Naive Semantic Entropy = {sem_entropy_naive:.4f}',
                                      f'Naive_log Semantic Entropy = {sem_entropy_naive_log:.4f}'],
                                b_plt_show=False,
                                save_path=os.path.join(show_save_path_ccid, f'rn152-rel-{ly_name}_{ccid}_naive_entropy.png'))
                    sem_probs_naive_dict_sorted_pd = pd.DataFrame(sem_probs_naive_dict_sorted.items(), columns=['name', 'prob'])
                    sem_probs_naive_dict_sorted_pd.to_csv(os.path.join(show_save_path_ccid, f'rn152-rel-{ly_name}_{ccid}_naive_entropy.csv'))

                    # clustered
                    sem_probs_cluster_dict = sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_cluster_dict']
                    sem_probs_cluster_dict_sorted = dict(sorted(sem_probs_cluster_dict.items(), key=lambda item: item[1]))
                    sem_clusters_probs = list(sem_probs_cluster_dict_sorted.values())
                    sem_clusters_names = list(sem_probs_cluster_dict_sorted.keys())
                    sem_entropy_clusters = semantic_entropys[ly_name][ccid]['sem_entropy_clusters']
                    sem_entropy_clusters_log = semantic_entropys[ly_name][ccid]['sem_entropy_clusters_log']
                    plt_entropy(probs=sem_clusters_probs, names=sem_clusters_names,
                                # title='Probability of Clustered Semantics',
                                xlabel='Semantic Name', ylabel='Probability of Semantic',
                                text=[f'Clustered Semantic Entropy = {sem_entropy_clusters:.4f}',
                                      f'Clustered_log Semantic Entropy = {sem_entropy_clusters_log:.4f}'],
                                b_plt_show=True,
                                save_path=os.path.join(show_save_path_ccid, f'rn152-rel-{ly_name}_{ccid}_cluster_entropy.png'))
                    sem_probs_cluster_dict_sorted_pd = pd.DataFrame(sem_probs_cluster_dict_sorted.items(), columns=['name', 'prob'])
                    sem_probs_cluster_dict_sorted_pd.to_csv(os.path.join(show_save_path_ccid, f'rn152-rel-{ly_name}_{ccid}_cluster_entropy.csv'))

                    # completed and clustered
                    sem_probs_cluster_pad_dict = sem_ccids_atoms_probs_dict[ly_name][ccid]['sem_probs_cluster_pad_dict']
                    sem_probs_cluster_pad_dict_sorted = dict(sorted(sem_probs_cluster_pad_dict.items(), key=lambda item: item[1]))
                    sem_cluster_pad_probs = list(sem_probs_cluster_pad_dict_sorted.values())
                    sem_cluster_pad_names = list(sem_probs_cluster_pad_dict_sorted.keys())
                    sem_entropy_clusters_pad = semantic_entropys[ly_name][ccid]['sem_entropy_clusters_pad']
                    sem_entropy_clusters_pad_log = semantic_entropys[ly_name][ccid]['sem_entropy_clusters_pad_log']
                    plt_entropy(probs=sem_cluster_pad_probs, names=sem_cluster_pad_names,
                                # title='Probability of Semantics',
                                xlabel='Semantic Name', ylabel='Probability of Semantic',
                                text=[f'Clustered Pad Semantic Entropy = {sem_entropy_clusters_pad:.4f}',
                                      f'Clustered_log Pad Semantic Entropy = {sem_entropy_clusters_pad_log:.4f}'],
                                b_plt_show=True,
                                save_path=os.path.join(show_save_path_ccid, f'rn152-rel-{ly_name}_{ccid}_cluster_pad_entropy.png'))
                    sem_probs_cluster_pad_dict_sorted_pd = pd.DataFrame(sem_probs_cluster_pad_dict_sorted.items(), columns=['name', 'prob'])
                    sem_probs_cluster_pad_dict_sorted_pd.to_csv(os.path.join(show_save_path_ccid, f'rn152-rel-{ly_name}_{ccid}_cluster_pad_entropy.csv'))

    return sem_desc, semantic_entropys, static_entropy_all_dict, static_entropy_avg_dict, sem_ccids_atoms_probs_dict





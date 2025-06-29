# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2021- 10- 19
Authors: Yu wenlong  and  DRAGON_501

*********************************************************************************
********************************Fighting! GO GO GO*******************************
*************************************Import***********************************"""

import time
import errno
import numpy as np

import csv
import torch

import matplotlib.pyplot as plt

from PIL import Image

import os

import warnings
warnings.filterwarnings("ignore")

torch.set_printoptions(profile="full", sci_mode=False)
now = int(time.time())
timeArray = time.localtime(now)
Time = time.strftime("%Y%m%d_%H%M", timeArray)


np.set_printoptions(suppress=True)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


"""********************************Variable***********************************"""
"""***************************************************************************"""
'''***************************************************************************'''


def plot_image(ax, img, norm):
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.astype('uint8')
    ax.imshow(img)


def makedirsExist(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
        else:
            raise


def clear_plot():
    plt.clf()
    plt.cla()
    plt.close()


def plot_func_all(args):
    pass


def read_csv(filename):
    data_dict = {}
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        num_cols = len(headers)

        for header in headers:
            data_dict[header] = []

        for row in reader:
            for i in range(num_cols):
                data_dict[headers[i]].append(row[i])

    return data_dict


def plt_show_save(image, b_show=False, path='', name='', title='', dpi=None):
    if dpi is not None:
        plt.figure(dpi=dpi)
    else:
        plt.figure()
    # plt.subplots_adjust(left=0, right=0, top=0.9, bottom=0)
    if isinstance(image, str):
        if os.path.isfile(image):
            image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    if image.shape[0] == 3:
        plt.imshow(np.transpose(image, (1, 2, 0)))
    else:
        plt.imshow(image)

    plt.grid(False)
    plt.axis('off')
    plt.title(title, pad=1)

    if path != '':
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, name), bbox_inches='tight', pad_inches=0, dpi=dpi)
    if b_show is True:
        plt.show(block=True)
    clear_plot()















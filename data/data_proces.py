# -*- coding:utf-8 -*-
"""*****************************************************************************

*************************************Import***********************************"""
import os
# from timm.data import ImageDataset
import torch.utils.data as data
import torchvision
from PIL import Image
"""**********************************Import***********************************"""
'''***************************************************************************'''


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):

    return pil_loader(path)


class MakeDataset(data.Dataset):

    def __init__(self,
                 args,
                 tokenizer=None,
                 model=None,
                 preprocess=None,
                 device=None,
                 label_root=None,
                 transform=None,
                 target_transform=None,
                 dataset_split='train',
                 loader=default_loader):
        root = args.data_dir
        if 'imagenet' in args.data_dir:
            dataset_split = torchvision.datasets.ImageNet(root, transform=transform, split=dataset_split)

            self.dataset_split = dataset_split

        if 'clip' in args.model_name and tokenizer is not None:
            self.b_clip = True
            self.model = model
            self.tokenizer = tokenizer
            self.preprocess = preprocess

            IN_dataset_source_split = dataset_split
            in_classes_list = IN_dataset_source_split.classes
            self.in_classes_list = [f"A photo of a {label[0]}" for label in in_classes_list]
            self.clip_classes = tokenizer(self.in_classes_list).to(device)  # torch.Size([1000, 77])

            print(f"Clip setting done, text_features shape: [1000, 1024]")
        else:
            self.b_clip = False
            self.tokenizer = None
            self.model = None
            self.preprocess = None
            self.clip_classes = None
            self.text_features = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.dataset_split[index]

        return img, target

    def select_idx(self, img_path):

        for idx, (i_path, i_target) in enumerate(self.dataset_split.imgs):
            if os.path.samefile(i_path, img_path) is True:
                return idx
            else:
                if idx == len(self.dataset_split.imgs) - 1:
                    return None

    def __len__(self):
        return len(self.dataset_split.imgs)

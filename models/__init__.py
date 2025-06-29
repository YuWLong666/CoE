# -*- coding:utf-8 -*-
"""*****************************************************************************
Time:    2022- 09- 22
Authors: Yu wenlong  and  DRAGON_501

*************************************Import***********************************"""
import torch

# ************************************************************************************************


# Note: You need to change the ckpt_path in the following function to customize your own CoE case.
def build_models(args, model_name=None, resume=None):

    msg = None
    if model_name is not None:
        pass
    else:
        model_name = args.model_name

    if resume is not None:
        pass
    else:
        resume = args.resume

    if model_name == 'rn50':
        from torchvision.models.resnet import resnet50
        model = resnet50(pretrained=True)

        # Load your own trained resnet
        if resume is not None:
            assert resume.endswith('.pth'), 'The resume file must be a .pth file.'

            model_ckpt = torch.load(resume, map_location='cpu')

            new_state_dict = {}
            for k, v in model_ckpt['state_dict'].items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print('load ckpt from {}'.format(resume))
            print('loaded ckpt msg: {}'.format(msg))

        criterion = None
        preprocess = None
        tokenizer = None

    elif model_name == 'rn152' and 'imagenet' in args.dataset:
        from torchvision.models.resnet import resnet152
        model = resnet152(pretrained=True)
        criterion = None
        preprocess = None
        tokenizer = None

    elif model_name == 'rn50_clip':
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='./model/resnet/rn50_clip/RN50.pt')
        tokenizer = open_clip.get_tokenizer('RN50')
        criterion = None

    elif model_name == 'vitb16_clip':
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/16",
                    pretrained='./output/pretrained/vit/xai_vit_b16_imagenet-val_CRP_ckptdownload/ViT-B-16.pt')

        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        criterion = None

    else:
        return None, None, None, None, None

    return model, criterion, preprocess, tokenizer, msg

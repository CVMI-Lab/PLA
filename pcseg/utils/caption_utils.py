import os
import json
import torch
import numpy as np

from . import commu_utils
from ..config import cfg


def get_caption_batch(caption_cfg, text_cfg, batch_dict, text_encoder):
    caption_infos = {}
    caption_data = batch_dict['caption_data']

    num_captions = 0
    for key in caption_cfg:
        if key in caption_cfg['KEY'] and caption_cfg[key].ENABLED:
            caption, idx = caption_data[key.lower()]['caption'], caption_data[key.lower()]['idx']
            num_captions += len(caption)

            # caption_embed: (K, 512), caption_idx: (N), (N > K)
            caption_embed, caption_idx = extract_caption_embed(caption, caption_cfg[key], text_cfg, text_encoder, cfg.LOCAL_RANK)
            normed_caption_embed = torch.nn.functional.normalize(caption_embed, dim=-1)

            caption_infos['caption_{}'.format(key.lower())] = {
                'caption_embed': normed_caption_embed, 'caption_idx': caption_idx, 'select_image_corr': idx
            }

    batch_dict['caption_infos'] = caption_infos
    batch_dict['num_caption'] = num_captions / batch_dict['batch_size']
    return batch_dict


def extract_caption_embed(image_captions, caption_cfg, text_cfg, text_encoder, rank):
    # (B*K, 512)

    if caption_cfg.get('GATHER_CAPTION', True):
        image_captions_list = commu_utils.all_gather(image_captions)
        image_captions_all = [jj for ii in image_captions_list for jj in ii]
        num_caption_list = [len(ii) for ii in image_captions_list]
    else:
        image_captions_all = image_captions
        num_caption_list = [0] * 100
        num_caption_list[rank] = len(image_captions_all)
    caption_embed_all = forward_text_encoder(image_captions_all, text_encoder)

    # remove duplicate captions and re-index them
    if text_cfg.get('REMOVE_DUPLICATE_CAPTIONS', True):
        num_caption_list = torch.LongTensor([0] + num_caption_list).cuda()
        idx = torch.arange(num_caption_list[rank + 1]).long().cuda() + torch.cumsum(num_caption_list, 0)[rank]
        caption_embeds, unique_indices = torch.unique(caption_embed_all, dim=0, return_inverse=True)
        caption_idx = unique_indices[idx]
    else:
        caption_embeds = caption_embed_all
        caption_idx = torch.arange(caption_embed_all.shape[0]).long().cuda()

    return caption_embeds, caption_idx


def forward_text_encoder(image_captions, text_encoder):
    with torch.no_grad():
        if len(image_captions) > 0:
            if cfg.MODEL.TASK_HEAD.TEXT_EMBED.NAME == 'CLIP':
                text_tokens = text_encoder.tokenizer(image_captions, truncate=True).cuda()
                text_embed = text_encoder.encode_text(text_tokens).float()
            elif cfg.MODEL.TASK_HEAD.TEXT_EMBED.NAME == 'Bert':
                text_tokens = text_encoder.tokenizer(image_captions, return_tensors="pt", padding=True).to('cuda')
                text_embed = text_encoder(**text_tokens).pooler_output
            else:
                raise NotImplementedError
        else:
            text_embed = torch.zeros((0, cfg.MODEL.TASK_HEAD.TEXT_EMBED.CHANNEL), dtype=torch.float32).cuda()
    return text_embed


def select_images(caption_cfg, image_name, image_corr):
    """
    TODO: put this part into dataset
    Select part of images for training 
    """
    batch_size = len(image_name)
    if caption_cfg.get('SAMPLE', 1) > 1:
        random_start = np.random.randint(caption_cfg.SAMPLE)
        image_name = [(np.array(image_name[i])[random_start::caption_cfg.SAMPLE]).tolist() for i in range(batch_size)]
        image_corr = [(np.array(image_corr[i], dtype=object)[random_start::caption_cfg.SAMPLE]).tolist() for i in range(batch_size)]
    if caption_cfg.SELECT == 'ratio' and caption_cfg.RATIO == 1.0:
        return image_name, image_corr

    selected_image_name = []
    selected_image_corr = []

    for i in range(batch_size):
        if image_name[i] is None or len(image_name[i]) == 0:  # lack 2d data
            selected_image_name.append([])
            selected_image_corr.append([])
            selected_idx = None
        elif caption_cfg.SELECT == 'fixed':
            # view-level caotion: random select fixed number
            num = int(caption_cfg.NUM)
            selected_idx = np.random.choice(len(image_name[i]), min(num, len(image_name[i])), replace=False)
        elif caption_cfg.SELECT == 'ratio':
            # sequence slicing
            ratio = caption_cfg.RATIO
            selected_idx = np.random.choice(len(image_name[i]), max(1, int(len(image_name[i]) * ratio)), replace=False)
        elif caption_cfg.SELECT == 'hybrid':
            num = max(int(caption_cfg.NUM), int(len(image_name[i]) * caption_cfg.RATIO))
            selected_idx = np.random.choice(len(image_name[i]), min(max(1, num), len(image_name[i])), replace=False)
        else:
            raise NotImplementedError

        if selected_idx is not None:
            selected_image_name.append(np.array(image_name[i])[selected_idx].tolist())
            selected_image_corr.append(
                np.array(image_corr[i], dtype=object)[selected_idx].tolist()
            )

    return selected_image_name, selected_image_corr


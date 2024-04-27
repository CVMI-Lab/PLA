import os
import json
import torch
import numpy as np

from . import commu_utils
from ..config import cfg


def only_keep_label_names(label_names, caption, idx):
    new_caption = []
    new_idx = []

    caption_counter = 0
    for b in range(len(idx)):
        new_idx.append([])
        for i in range(len(idx[b])):
            c = caption[caption_counter]
            if c in label_names:
                new_caption.append(c)
                new_idx[b].append(idx[b][i])
            caption_counter += 1
    return new_caption, new_idx


def get_caption_batch(caption_cfg, text_cfg, batch_dict, text_encoder):
    caption_infos = {}
    caption_data = batch_dict['caption_data']

    num_captions = 0
    for key in caption_cfg:
        if key in caption_cfg['KEY'] and caption_cfg[key].ENABLED:
            caption, idx = caption_data[key.lower()]['caption'], caption_data[key.lower()]['idx']
            if caption_cfg[key].get('ONLY_LABEL_NAMES', False):
                caption, idx = only_keep_label_names(caption_cfg[key].LABEL_NAMES, caption, idx)
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
    if caption_cfg.get('GATHER_CAP_MODE', 'cap') == 'cap' and caption_cfg.get('GATHER_CAPTION', True):
        image_captions_all, num_caption_list = gather_raw_captions(image_captions)
    else:
        image_captions_all = image_captions
        num_caption_list = [0] * 100
        num_caption_list[rank] = len(image_captions_all)
    caption_embed_all = forward_text_encoder(image_captions_all, text_encoder)

    # remove duplicate captions and re-index them
    if text_cfg.get('REMOVE_DUPLICATE_CAPTIONS', True):
        if caption_cfg.get('GATHER_CAP_MODE', 'cap') == 'emb' and caption_cfg.get('GATHER_CAPTION', True):
            caption_embed_all, num_caption_list = gather_caption_embs(caption_cfg, rank, caption_embed_all, image_captions)
        num_caption_list = torch.LongTensor([0] + num_caption_list).cuda()
        idx = torch.arange(num_caption_list[rank + 1]).long().cuda() + torch.cumsum(num_caption_list, 0)[rank]
        caption_embeds, unique_indices = torch.unique(caption_embed_all, dim=0, return_inverse=True)
        caption_idx = unique_indices[idx]
    else:
        caption_embeds = caption_embed_all
        caption_idx = torch.arange(caption_embed_all.shape[0]).long().cuda()

    if caption_cfg.get('WHOLE_LABEL_NAMES', False):
        label_names = caption_cfg.LABEL_NAMES
        other_label_names = set(label_names) - set(image_captions_all)
        caption_embed_others = forward_text_encoder(list(other_label_names), text_encoder)
        caption_embeds = torch.cat((caption_embeds, caption_embed_others), dim=0)

    return caption_embeds, caption_idx


def gather_caption_embs(caption_cfg, rank, caption_embed, image_captions):
    if caption_cfg.get('GATHER_CAPTION', True):
        caption_embed_all, num_caption_list = commu_utils.all_gather_with_count(caption_embed)
    else:
        caption_embed_all = caption_embed
        num_caption_list = [0] * commu_utils.get_world_size()
        num_caption_list[rank] = len(image_captions)

    return caption_embed_all, num_caption_list

def gather_raw_captions(image_captions):
    image_captions_list = commu_utils.all_gather(image_captions)
    image_captions_all = [jj for ii in image_captions_list for jj in ii]
    num_caption_list = [len(ii) for ii in image_captions_list]
    
    return image_captions_all, num_caption_list


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


def enlarge_boxes_size(origin_boxes, enlarge_ratio, max_box_size, image_shape):
    """
    Args:
        origin_boxes: [N, 4]. [y_min, x_min, y_max, x_max]
        enlarge_ratio: scalar. Enlarge ratio for the origin boxes
        max_box_size:
        image_shape: (2): height, width

    Returns:

    """
    image_height, image_width = image_shape[0], image_shape[1]
    boxes_size = (origin_boxes[:, 2] - origin_boxes[:, 0]) * (origin_boxes[:, 3] - origin_boxes[:, 1])
    # figure out the boxes that need to be enlarge
    enlarge_boxes_mask = boxes_size < max_box_size
    enlarge_boxes = origin_boxes[enlarge_boxes_mask]
    width = enlarge_boxes[:, 3] - enlarge_boxes[:, 1]
    height = enlarge_boxes[:, 2] - enlarge_boxes[:, 0]
    width_enlarge = width * (enlarge_ratio - 1)
    height_enlarge = height * (enlarge_ratio - 1)
    enlarge_boxes[:, 0] = np.maximum(0, enlarge_boxes[:, 0] - height_enlarge / 2)
    enlarge_boxes[:, 2] = np.minimum(image_height - 1, enlarge_boxes[:, 2] + height_enlarge / 2)
    enlarge_boxes[:, 1] = np.maximum(0, enlarge_boxes[:, 1] - width_enlarge / 2)
    enlarge_boxes[:, 3] = np.minimum(image_width - 1, enlarge_boxes[:, 3] + width_enlarge / 2)

    origin_boxes[enlarge_boxes_mask] = enlarge_boxes
    return origin_boxes


def get_sliding_windows(image_shape, window_size, strides):
    height, width, channel = image_shape
    box_list = []  # each box should in format [y_min, x_min, y_max, x_max]

    sampling_row_coord = list(np.arange(0, height - window_size[0] - 1, strides[0]))
    sampling_col_coord = list(np.arange(0, width - window_size[1] - 1, strides[1]))

    if height - sampling_row_coord[-1] - window_size[0] > window_size[0] / 2:
        sampling_row_coord.append(height - window_size[0] - 1)

    if width - sampling_col_coord[-1] - window_size[1] > window_size[1] / 2:
        sampling_col_coord.append(width - window_size[1] - 1)

    for row in sampling_row_coord:
        for col in sampling_col_coord:
            box = (row, col, row + window_size[0], col + window_size[1])
            box_list.append(box)

    boxes = np.array(box_list)
    return boxes


def n_captions_for_points(image_corr_dict, n_points):
    n_captions_points = np.zeros(n_points, dtype=np.int32)
    for caption_type, caption_corr_list in image_corr_dict.items():
        for caption_corr in caption_corr_list:
            n_captions_points[caption_corr] += 1

    return n_captions_points

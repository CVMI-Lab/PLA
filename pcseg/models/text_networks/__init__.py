import os
import logging
import torch

from . import text_models
from .prompt_template import template_meta
from ...config import cfg


def build_text_model(model_cfg):
    tokenizer, text_encoder = getattr(
        text_models, f'get_{model_cfg.NAME.lower()}_model'
    )(model_cfg.BACKBONE)

    text_encoder.tokenizer = tokenizer
    return text_encoder


def load_text_embedding_from_path(text_emb_cfg):
    text_emb_path = os.path.join(cfg.DATA_CONFIG.DATA_PATH, text_emb_cfg.PATH)
    text_embedding = torch.load(text_emb_path, map_location=torch.device('cpu')).detach()
    if text_emb_cfg.get('NORM', True):
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    print("=> loaded text embedding from path '{}'".format(text_emb_path))
    return text_embedding


def is_bg_class(c):
    return (c.lower() == 'wall') or (c.lower() == 'floor') or (c.lower() == 'ceiling') or (c.lower() =='otherfurniture')


def build_text_token_from_class_names(model_cfg, class_names):
    if model_cfg.TEMPLATE == 'lseg':  # only instance classes are encoded with prompt
        return [template_meta[model_cfg.TEMPLATE][0].format(c) if not is_bg_class(c) else c for c in class_names]
    else:
        return [template_meta[model_cfg.TEMPLATE][0].format(c) for c in class_names]


def load_text_embedding_from_encoder(model_cfg, text_encoder, logger=logging.getLogger()):
    text_encoder.cuda()
    class_names = cfg.TEXT_ENCODER.CATEGORY_NAMES
    text = build_text_token_from_class_names(model_cfg, class_names)

    if model_cfg.NAME == 'CLIP':
        text_tokens = text_encoder.tokenizer(text).cuda()
        text_embedding = text_encoder.encode_text(text_tokens)
    elif model_cfg.NAME == 'BERT':
        text_tokens = text_encoder.tokenizer(text, return_tensors="pt", padding=True).to('cuda')
        text_embedding = text_encoder(**text_tokens).pooler_output
    else:
        raise NotImplementedError

    if model_cfg.get('NORM', True):
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    logger.info("=> loaded text embedding from '{}'".format(model_cfg.NAME))
    return text_embedding.detach().cpu()

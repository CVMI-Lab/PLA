import os
import clip
import torch
from transformers import AutoTokenizer, AutoModel

from pcseg.models.text_networks.text_models import get_clip_model


class_names = {
    'scannet': ['wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'showercurtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'],
    's3dis': ['ceiling', 'floor', 'wall', 'beam', 'column',
              'window', 'door', 'table', 'chair', 'sofa',
              'bookcase', 'board', 'clutter']
}


def construct_input_from_class_name(input, tokenizer):
    inputs = tokenizer(input, return_tensors="pt", padding=True)
    return inputs


def get_embedding(args):
    if args.model.startswith('clip'):
        backbone_name = args.model[5:]
        input = class_names[args.dataset]
        _, model = get_clip_model(backbone_name)
        model = model.cuda()
        text = clip.tokenize(input).cuda()
        output = model.encode_text(text)
        print(output.shape)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)

        inputs = construct_input_from_class_name(class_names[args.dataset], tokenizer)
        outputs = model(**inputs)
        output = outputs.pooler_output
        print(outputs.pooler_output.shape)

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('language model')
    parser.add_argument('--model', default='clip-ViT-B/16', type=str, help='language model name')
    parser.add_argument('--dataset_path', default='../data/scannetv2', type=str, help='language model name')
    parser.add_argument('--dataset', default='scannet', type=str, help='dataset name')
    args = parser.parse_args()

    category_embedding = get_embedding(args)

    file_name = '{}_{}_{}_text_embed.pth'.format(
        args.dataset, len(class_names[args.dataset]), args.model.replace('/', '')
    )
    save_dir = os.path.join(args.dataset_path, 'text_embed')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)

    torch.save(category_embedding, save_path)
    print("Saving category embedding into: ", save_path)

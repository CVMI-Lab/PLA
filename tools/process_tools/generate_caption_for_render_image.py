import json
import argparse
import glob
import os
import torch
import tqdm

from tools.process_tools.generate_caption import init_model


def get_caption_and_write(save_caption_path, image_path_list):
    image_name_list = []
    caption_list = []
    print('Start to process image captions.....')
    image_path_chunks = [image_path_list[x:x + 20] for x in range(0, len(image_path_list), 20)]
    for image_path_chunk in tqdm.tqdm(image_path_chunks, total=len(image_path_chunks)):
        res = caption_model.predict_step(image_path_chunk)
        image_name_list.extend(list(res.keys()))
        caption_list.extend(list(res.values()))

    with open(save_caption_path, 'w') as f:
        for i, image_name in enumerate(image_name_list):
            text = f'{image_name}:\t{caption_list[i]}\n'
            f.write(text)

    print(f'Successfully write to {save_caption_path} .....')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('language model')
    parser.add_argument('--caption_model', default='nlpconnect/vit-gpt2-image-captioning',
                        choices=['nlpconnect/vit-gpt2-image-captioning', 'damo/ofa_image-caption_coco_large_en'],
                        type=str, help='language model name')
    parser.add_argument('--max_length', default=64, type=int, help='max length')
    parser.add_argument('--data_render_path', default='/data/datasets/scannetv2/241_random/images/', type=str, help='')
    parser.add_argument('--data_gt_path', default='/data/datasets/scannetv2/241_random/images/', type=str,
                        help='language model name')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use device: {device}')

    # pred_image_path = sorted(glob.glob(os.path.join(args.data_render_path, 'step*-coarse*.png')))
    # gt_image_path = sorted(glob.glob(os.path.join(args.data_gt_path, 'step*-gt*.png')))
    pred_image_path = sorted(glob.glob(os.path.join(args.data_render_path, '*.jpg')))
    gt_image_path = sorted(glob.glob(os.path.join(args.data_gt_path, '*.png')))

    caption_model = init_model(args.caption_model, device, max_length=args.max_length)

    # save pred image
    pred_caption_path = os.path.join(args.data_render_path, 'pointnerf_caption.txt')
    gt_caption_path = os.path.join(args.data_gt_path, 'gt_caption.txt')

    print('Start to process PointNeRF image captions.....')
    get_caption_and_write(pred_caption_path, pred_image_path)

    print('Start to process real image captions.....')
    get_caption_and_write(gt_caption_path, gt_image_path)


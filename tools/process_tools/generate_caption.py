import json
import argparse

import matplotlib.pyplot as plt
import torch
import glob
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from tqdm import tqdm

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from torchvision import transforms

from pcseg.utils import common_utils

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class ViT_GPT2(object):
    def __init__(self, name, device, max_length, **kwargs):
        self.model = VisionEncoderDecoderModel.from_pretrained(name, local_files_only=False)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(name, local_files_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=False)
        self.model.to(device)
        # self.feature_extractor.to(device)
        self.device = device
        self.max_length = max_length
        self.num_beams = 4

    def predict_step(self, image_paths, image_name_list=None):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        res = {}
        for idx, image_path in enumerate(image_paths):
            if image_name_list is None:
                image_name = image_path.split('/')[-1].split('.')[0]
            else:
                image_name = image_name_list[idx].lower()
            res[image_name] = preds[idx]
        return res

    def predict_step_with_image(self, images, image_name_list=None):
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        res = {}
        for idx, image_name in enumerate(image_name_list):
            image_name = image_name.lower()
            res[image_name] = preds[idx]

        # visualization debug code
        # import tools.visual_utils.open3d_vis_utils as vis
        # for idx, image in enumerate(images):
        #     image = np.array(image)
            # plt.imsave('vis_output/' + image_name_list[idx] + '_raw.png', image / 255.0)
            # vis.plot_image_with_caption(image / 255.0, res[image_name_list[idx]], image_name_list[idx])

        return res


def init_model(name, device, **kwargs):
    zoo = {
        'nlpconnect/vit-gpt2-image-captioning': ViT_GPT2,
        # 'damo/ofa_image-caption_coco_large_en': OFA
    }
    return zoo[name](name, device, **kwargs)


def init_summarizer(args, device):
    from transformers import pipeline as sum_pipeline
    summarizer = sum_pipeline("summarization", model=args.summarizer, device=0 if device.type=='cuda' else -1)
    return summarizer


def write_caption_to_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'The caption is dump to {path}')


class ProcessorTemplate(object):
    def __init__(self, device):
        self.model = init_model(args.caption_model, device, max_length=args.max_length)
        if args.caption_mode == 'scene_caption':
            self.summarizer = init_summarizer(args, device)

    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = np.array(image)
        return image

    @staticmethod
    def extract_entity(view_caption):
        caption_entity = {}
        for scene in view_caption:
            caption_entity[scene] = {}
            for frame in view_caption[scene]:
                caption = view_caption[scene][frame]
                tokens = nltk.word_tokenize(caption)
                tagged = nltk.pos_tag(tokens)
                entities = []
                # entities = nltk.chunk.ne_chunk(tagged)
                for e in tagged:
                    if e[1].startswith('NN'):
                        entities.append(e[0])
                new_caption = ' '.join(entities)
                caption_entity[scene][frame] = new_caption
        return caption_entity

    @staticmethod
    def compute_intersect_and_diff(c1, c2):
        old = set(c1) - set(c2)
        new = set(c2) - set(c1)
        intersect = set(c1) & set(c2)
        return old, new, intersect


class ScanNetProcessor(ProcessorTemplate):
    def __init__(self, device):
        super(ScanNetProcessor, self).__init__(device)
        # self.scene_path = sorted(glob.glob(os.path.join(args.dataset_path, args.image_tag, 'scene*')))
        with open(os.path.join(args.dataset_path, 'scannetv2_{}.txt'.format(args.dataset_split))) as fin:
            self.scene_list = fin.readlines()
        self.scene_list = sorted([s.strip() for s in self.scene_list])[:10]

    def process_view_caption(self):
        captions_view = {}

        print('Processing view captions.....')
        for scene_name in tqdm(self.scene_list):
            # scene_name = scene.split('/')[-1]
            img_path = sorted(glob.glob(
                os.path.join(args.dataset_path, args.image_tag, '{}/color/*.jpg'.format(scene_name))))
            res = self.model.predict_step(img_path)
            captions_view[scene_name] = res

        write_caption_to_file(
            captions_view,
            os.path.join(args.output_dir, 'caption_view_{}_{}_{}.json'.format(
                args.dataset, args.caption_model.split('/')[-1], args.tag))
        )

    def process_scene_caption(self):
        print('Processing scene captions.....')

        # load view caption
        caption_view_path = args.view_caption_path
        # os.path.join(args.output_dir, 'caption_view_{}_{}_{}.json'.format(
        #     args.dataset, args.caption_model.split('/')[-1], args.tag))
        captions_view = json.load(open(caption_view_path, 'r'))
        print(f'load view captions from {caption_view_path}')
        captions_scene = {}

        for i, scene in tqdm(enumerate(self.scene_list)):
            # scene = scene.split('/')[-1]
            text = '. '.join(captions_view[scene].values())
            if len(text.split(' ')) > 75:
                sum_caption = self.summarizer(text, max_length=75)[0]['summary_text']
            else:
                sum_caption = text
            captions_scene[scene] = sum_caption

        write_caption_to_file(
            captions_scene,
            os.path.join(args.output_dir, 'caption_scene_{}_{}_{}.json'.format(
                args.dataset, args.caption_model.split('/')[-1], args.tag))
        )

    def process_entity_caption(self):

        print('Processing entity captions.....')

        view_caption = json.load(open(args.view_caption_path, 'r'))
        view_caption_corr_idx = pickle.load(open(args.view_caption_corr_idx_path, 'rb'))
        # res = self.model.predict_step(img_path)
        view_entity_caption = self.extract_entity(view_caption)
        captions_entity = self.get_entity_caption(view_entity_caption, view_caption_corr_idx)
        write_caption_to_file(
            captions_entity,
            os.path.join(args.output_dir, 'caption_entity_{}_{}_{}.json'.format(
                args.dataset, args.caption_model.split('/')[-1], args.tag))
        )

    def get_entity_caption(self, view_entity_caption, view_caption_corr_idx):
        entity_caption = {}

        minpoint = 100
        ratio = args.entity_overlap_thr
        for scene in tqdm(self.scene_list):
            if scene not in view_caption_corr_idx:
                continue
            frame_idx = view_caption_corr_idx[scene]
            entity_caption[scene] = {}
            entity_num = 0
            frame_keys = list(frame_idx.keys())
            for ii in range(len(frame_keys) - 1):
                for jj in range(ii + 1, len(frame_keys)):
                    idx1 = frame_idx[frame_keys[ii]].cpu().numpy()
                    idx2 = frame_idx[frame_keys[jj]].cpu().numpy()
                    c = view_entity_caption[scene][frame_keys[ii]].split(' ')
                    c2 = view_entity_caption[scene][frame_keys[jj]].split(' ')
                    if 'room' in c:  # remove this sweeping word
                        c.remove('room')
                    if 'room' in c2:
                        c2.remove('room')

                    old, new, intersection = self.compute_intersect_and_diff(idx1, idx2)
                    old_c, new_c, intersection_c = self.compute_intersect_and_diff(c, c2)

                    if len(intersection) > minpoint and len(intersection_c) > 0 and \
                        len(intersection) / float(min(len(idx1), len(idx2))) <= ratio:
                        entity_caption[scene]['entity_{}'.format(entity_num)] = ' '.join(list(intersection_c))
                        # entity_caption_corr_idx[scene]['entity_{}'.format(entity_num)] = torch.IntTensor(list(intersection))
                        entity_num += 1
                    if len(old) > minpoint and len(old_c) > 0 and len(old) / float(len(idx1)) <= ratio:
                        entity_caption[scene]['entity_{}'.format(entity_num)] = ' '.join(list(old_c))
                        # entity_caption_corr_idx[scene]['entity_{}'.format(entity_num)] = torch.IntTensor(list(old))
                        entity_num += 1
                    if len(new) > minpoint and len(new_c) > 0 and len(new) / float(len(idx2)) <= ratio:
                        entity_caption[scene]['entity_{}'.format(entity_num)] = ' '.join(list(new_c))
                        # entity_caption_corr_idx[scene]['entity_{}'.format(entity_num)] = torch.IntTensor(list(new))
                        entity_num += 1

        return entity_caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser('language model')
    parser.add_argument('--caption_model', default='nlpconnect/vit-gpt2-image-captioning',
                        choices=['nlpconnect/vit-gpt2-image-captioning'],
                        type=str, help='language model name')
    parser.add_argument('--summarizer', default='facebook/bart-large-cnn', type=str, help='language model name')
    parser.add_argument('--dataset', default='scannet', type=str, help='dataset name')
    parser.add_argument('--output_dir', required=True, help='path to output folder')
    parser.add_argument('--max_length', default=64, type=int, help='max length')
    parser.add_argument('--caption_mode', default='view_caption',
                        choices=['view_caption', 'scene_caption', 'entity_caption'])
    parser.add_argument('--dataset_path', default='./data/scannetv2', type=str, help='language model name')
    parser.add_argument('--tag', default='', type=str, help='')

    # OFA
    # parser.add_argument('--ofa_ckpt_dir', default='', type=str, help='')

    # for speed up
    parser.add_argument('--cur', default=1, type=int, help='')
    parser.add_argument('--all', default=1, type=int, help='')

    # scannet args
    parser.add_argument('--image_tag', default='scannet_frames_25k', type=str, help='')

    # entity caption, scene caption
    parser.add_argument('--entity_overlap_thr', default=0.3, help='threshold ratio for filtering out large entity-level point set')
    parser.add_argument('--view_caption_path', default=None, help='path for view-level caption')
    parser.add_argument('--view_caption_corr_idx_path', default=None, help='path for view-level caption corresponding index')

    # To split into different
    parser.add_argument('--split_num', default=-1, type=int, help='')
    parser.add_argument('--split_total', default=-1, type=int, help='')

    # 
    parser.add_argument('--dataset_split', default='train', help='train / val')
    global args
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'scannet':
        processor = ScanNetProcessor(device)
    elif args.dataset == 's3dis':
        # TODO: to support S3DIS generate caption
        raise NotImplementedError
    else:
        raise NotImplementedError

    if args.caption_mode == 'view_caption':
        """
        python -m tools.process_tools.generate_caption --dataset scannet \
        --caption_mode view_caption \
        --output_dir ./data/scannetv2/text_embed
        """
        processor.process_view_caption()
    elif args.caption_mode == 'scene_caption':
        """
        python -m tools.process_tools.generate_caption --dataset scannet \
        --caption_mode scene_caption \
        --output_dir ./data/scannetv2/text_embed
        """
        processor.process_scene_caption()
    elif args.caption_mode == 'entity_caption':
        """
        python -m tools.process_tools.generate_caption --dataset scannet \
        --caption_mode entity_caption \
        --output_dir ./data/scannetv2/text_embed \
        --view_caption_path ./data/scannetv2/text_embed/caption_view_scannet_vit-gpt2-image-captioning_25k.json \
        --view_caption_corr_idx_path ./data/scannetv2/scannetv2_view_vit-gpt2_matching_idx.pickle
        """
        processor.process_entity_caption()
    else:
        raise NotImplementedError

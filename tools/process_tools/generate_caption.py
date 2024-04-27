import json
import argparse

import matplotlib.pyplot as plt
import torch
import glob
import os
import pickle
import numpy as np

from PIL import Image
from tqdm import tqdm

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq
from torchvision import transforms

from pcseg.utils import common_utils
from pcseg.utils.caption_utils import get_sliding_windows, enlarge_boxes_size


class ViT_GPT2(object):
    def __init__(self, name, device, max_length, **kwargs):
        self.model = VisionEncoderDecoderModel.from_pretrained(name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
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


class OFA(object):
    def __init__(self, name, device, **kwargs):
        from transformers import OFATokenizer, OFAModel
        from transformers.models.ofa.generate import sequence_generator

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.tokenizer = OFATokenizer.from_pretrained(args.ofa_ckpt_dir)
        self.model = OFAModel.from_pretrained(args.ofa_ckpt_dir, use_cache=True).to(device)
        self.inputs = self.tokenizer([" what does the image describe?"], return_tensors="pt").input_ids.to(device)

        self.generator = sequence_generator.SequenceGenerator(
            tokenizer=self.tokenizer,
            beam_size=5,
            max_len_b=16,
            min_len=0,
            no_repeat_ngram_size=3,
        ).to(device)
        self.device = device

    def predict_step(self, image_paths, image_name_list=None):
        caption_2d = {}
        for idx, image_path in enumerate(image_paths):
            if image_name_list is None:
                image_name = image_path.split('/')[-1][:-4]
            else:
                image_name = image_name_list[idx].lower()

            caption = self.model(image_path)[self.outputkey_caption]
            caption_2d[image_name] = caption[0]
        return caption_2d

    def predict_step_with_image(self, images, image_name_list=None):
        res = {}

        for idx, image in enumerate(images):
            caption = self.pred_single_image(image)
            res[image_name_list[idx]] = caption

        # import tools.visual_utils.open3d_vis_utils as vis
        # for idx, image in enumerate(images):
            # image = np.array(image)
            # plt.imsave('vis_output/' + image_name_list[idx] + '_raw.png', image / 255.0)
            # vis.plot_image_with_caption(image / 255.0, res[image_name_list[idx]], image_name_list[idx])

        return res

    def pred_single_image(self, image):
        patch_img = self.patch_resize_transform(image).unsqueeze(0).to(self.device)
        data = {}
        data["net_input"] = {
            "input_ids": self.inputs,
            'patch_images': patch_img,
            'patch_masks': torch.tensor([True]).to(self.device)
        }
        gen_output = self.generator.generate([self.model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
        caption = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

        return caption


class BLIP2(object):
    def __init__(self, name, device, **kwargs):
        from lavis.models import load_model_and_preprocess

        # model_name = 'blip2_t5'
        model_type = 'pretrain_flant5xl'

        print('>>> Initialize BLIP2 model....')
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name, model_type=model_type, is_eval=True, device=device
        )

    def predict_step(self, image_paths, image_name_list=None):
        caption_2d = {}
        for idx, image_path in enumerate(image_paths):
            if image_name_list is None:
                image_name = image_path.split('/')[-1][:-4]
            else:
                image_name = image_name_list[idx].lower()

            caption = self.predict_step_single(image_path)
            caption_2d[image_name] = caption
        return caption_2d

    def predict_step_single(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image})[0]
        return output

    def predict_step_with_image(self, images, image_name_list=None):
        res = {}

        for idx, image in enumerate(images):
            _image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            caption = self.model.generate({"image": _image})[0]
            res[image_name_list[idx]] = caption

        return res


def init_model(name, device, **kwargs):
    zoo = {
        'nlpconnect/vit-gpt2-image-captioning': ViT_GPT2,
        'damo/ofa_image-caption_coco_large_en': OFA,
        'blip2_t5': BLIP2
    }
    return zoo[name](name, device, **kwargs)


def init_summarizer(args, device):
    from transformers import pipeline as sum_pipeline
    summarizer = sum_pipeline("summarization", model=args.summarizer, device=0 if device.type=='cuda' else -1)
    return summarizer


def write_caption_to_file(data, path):
    if args.split_id != -1 and args.split_total != -1:
        path = path[:-5] + f'_part{args.split_id}' + path[-5:]

    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'The caption is dump to {path}')


class ProcessorTemplate(object):
    def __init__(self, device):
        self.model = init_model(args.caption_model, device, max_length=args.max_length)
        # self.summarizer = init_summarizer(args, device)

    @staticmethod
    def crop_image_with_boxes(image, boxes, min_image_area=None):
        cropped_image_list = []
        for box in boxes:
            y_min, x_min, y_max, x_max = box
            area = (y_max - y_min) * (x_max - x_min)
            if min_image_area and area < min_image_area:
                continue
            cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            crop = Image.fromarray(cropped_image.astype('uint8'), 'RGB')
            cropped_image_list.append(crop)

        return cropped_image_list

    @staticmethod
    def read_image(image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = np.array(image)
        return image


class ScanNetProcessor(ProcessorTemplate):
    def __init__(self, device):
        super(ScanNetProcessor, self).__init__(device)
        self.scene_path = sorted(glob.glob(os.path.join(args.dataset_path, args.image_tag, 'scene*')))

        with open(os.path.join('data/split_files', 'scannetv2_train.txt'), 'r') as fin:
            scene_name_list = [x.strip() for x in fin.readlines()]
            scene_name_list = sorted(scene_name_list)

        # filter scenes that are not for training
        self.scene_path = [x for x in self.scene_path if x.split('/')[-1] in scene_name_list]
        # try:
        #     self.detic_infos = json.load(open(args.detic_info, 'r'))
        # except:
        #     self.detic_infos = pickle.load(open(args.detic_info, 'rb'))

        if args.split_id != -1 and args.split_total != -1:
            assert args.split_id < args.split_total
            step = [x for x in range(0, len(self.scene_path), int(len(self.scene_path) / args.split_total) + 1)]
            step += [len(self.scene_path)]
            start_idx = max(0, step[args.split_id])
            end_idx = min(len(self.scene_path), step[args.split_id + 1])
            self.scene_path = self.scene_path[start_idx: end_idx]
            if hasattr(self, 'detic_infos'):
                self.detic_infos = self.detic_infos[start_idx:end_idx]

        print(f'Total {len(self.scene_path)} samples are loaded.')

    def process_view_caption(self):
        captions_view = {}

        print('Processing view captions.....')
        for scene in tqdm(self.scene_path):
            scene_name = scene.split('/')[-1]
            img_path = sorted(glob.glob('{}/color/*.jpg'.format(scene)))
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
        caption_view_path = os.path.join(args.output_dir, 'caption_view_{}_{}_{}.json'.format(
            args.dataset, args.caption_model.split('/')[-1], args.tag))
        captions_view = json.load(open(caption_view_path, 'r'))
        print(f'load view captions from {caption_view_path}')
        captions_scene = {}

        for i, scene in enumerate(self.scene_path):
            scene = scene.split('/')[-1]
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

    @staticmethod
    def process_detic_crop_caption_with_raw_pred():
        captions_crop = {}
        detic_infos = pickle.load(open(args.detic_info, 'rb'))

        print(f'Processing Detic raw prediction for ScanNet from {args.detic_info}')
        for idx, info in tqdm(enumerate(detic_infos), total=len(detic_infos)):
            scene_name = info['scene_name']
            image_infos = info['infos']
            scene_crop_caption = {}
            for image_name, image_info in image_infos.items():
                image_classes = image_info['classes']
                counter = 0
                for class_name in image_classes:
                    class_name = class_name.replace('_', ' ')
                    scene_crop_caption[f'{image_name.lower()}_{counter}'] = class_name
                    counter += 1

            captions_crop[scene_name] = scene_crop_caption

        write_caption_to_file(
            captions_crop,
            os.path.join(args.output_dir, 'caption_crop_{}_{}_{}.json'.format(
                args.dataset, args.caption_model.split('/')[-1], args.tag))
        )

    def process_basic_crop_caption_with_caption_idx_info(self):
        caption_basic_crop = {}
        window_size = np.array(args.window_size)
        strides = (int(window_size[0] * (1 - args.overlap_ratio)), int(window_size[1] * (1 - args.overlap_ratio)))

        print('Processing basic crop captions.....')
        for scene_path in tqdm(self.scene_path):
            scene_name = scene_path.split('/')[-1]
            img_path_list = sorted(glob.glob('{}/color/*.jpg'.format(scene_path)))
            # scene_caption_idx_info = caption_idx_info[scene_name]
            caption_basic_crop[scene_name] = {}
            for img_path in tqdm(img_path_list, total=len(img_path_list)):
                image = self.read_image(img_path)
                image_name = img_path.split('/')[-1].split('.')[0]

                # boxes = scene_caption_idx_info['boxes'][image_name.lower()]
                boxes = get_sliding_windows(image.shape, window_size, strides)

                cropped_images = self.crop_image_with_boxes(image, boxes)
                num_cropped_image = len(cropped_images)
                cropped_image_name_list = [f'{image_name.lower()}_{i}' for i in range(num_cropped_image)]
                res = self.model.predict_step_with_image(cropped_images, image_name_list=cropped_image_name_list)
                caption_basic_crop[scene_name].update(res)

        write_caption_to_file(
            caption_basic_crop,
            os.path.join(args.output_dir, 'caption_basic_crop_{}_{}_{}.json'.format(
                args.dataset, args.caption_model.split('/')[-1], args.tag))
        )

    def process_detic_crop_caption_with_caption(self):
        captions_detic_crop_cap = {}
        
        print(f'Processing detic boxes with image captions from {args.detic_info}')
        for idx ,scene_path in tqdm(enumerate(self.scene_path), total=len(self.scene_path)):
            scene_name = scene_path.split('/')[-1]
            img_path_list = sorted(glob.glob('{}/color/*.jpg'.format(scene_path)))
            scene_detic_info = self.detic_infos[idx]
            assert scene_detic_info['scene_name'] == scene_name
            captions_detic_crop_cap[scene_name] = {}
            for img_path in tqdm(img_path_list, total=len(img_path_list)):
                image = self.read_image(img_path)
                image_name = img_path.split('/')[-1].split('.')[0]

                if image_name in scene_detic_info['infos']:
                    boxes = scene_detic_info['infos'][image_name]['boxes']
                else:
                    # import ipdb; ipdb.set_trace(context=20)
                    # raise ValueError(f'split_id: {args.split_id}, idx: {idx}, image_name: {image_name.zfill(6)}')
                    continue
                     
                if boxes.shape[0] == 0:
                    continue
                if args.enlarge_box_ratio > 1.0:
                    boxes = enlarge_boxes_size(boxes, args.enlarge_box_ratio, args.enlarge_boxes_max_thresh, image.shape)

                cropped_images = self.crop_image_with_boxes(
                    image, boxes, min_image_area=args.enlarge_box_ratio * args.min_image_crop_area
                )
                num_cropped_image = len(cropped_images)
                if num_cropped_image == 0:
                    continue
                cropped_image_name_list = [f'{image_name.lower()}_{i}' for i in range(num_cropped_image)]
                res = self.model.predict_step_with_image(cropped_images, image_name_list=cropped_image_name_list)
                captions_detic_crop_cap[scene_name].update(res)

        caption_path = 'caption_detic_crop_cap_{}_{}_{}.json'.format(
            args.dataset, args.caption_model.split('/')[-1], args.tag
        )

        write_caption_to_file(captions_detic_crop_cap, os.path.join(args.output_dir, caption_path))

    def process_dense_caption(self):
        model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
        model.cuda()
        processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)

        prompt = "<grounding> Describe this image in detail:"

        print('Processing dense captions.....')
        captions_dict = {}
        caption_box_info = []
        for scene in tqdm(self.scene_path):
            scene_name = scene.split('/')[-1]
            captions_dict[scene_name], caption_box_info_single_scene = \
                self.process_dense_caption_single_scene(scene, model, processor, prompt)
            caption_box_info.append(
                {'infos': caption_box_info_single_scene, 'scene_name': scene_name})

        caption_path = os.path.join(args.output_dir, 'caption_dense_{}_{}_{}.json'.format(
                args.dataset, args.version, args.tag))
        write_caption_to_file(captions_dict, caption_path)

        caption_box_info_path = os.path.join(args.output_dir, 'caption_dense_box_info_{}_{}_{}.pkl'.format(
            args.dataset, args.version, args.tag))
        if args.split_id != -1 and args.split_total != -1:
            caption_box_info_path = caption_box_info_path[:-4] + f'_part{args.split_id}' + caption_box_info_path[-4:]
        with open(caption_box_info_path, 'wb') as f:
            pickle.dump(caption_box_info, f)

    def process_dense_caption_single_scene(self, scene, model, processor, prompt):
        img_path = sorted(glob.glob('{}/color/*.jpg'.format(scene)))

        captions_dict_single_scene = {}
        caption_box_info_single_scene = {}
        images = []
        for img_ in img_path:
            image_name = img_.split('/')[-1].split('.')[0]
            image = Image.open(img_)
            images.append(image)

        for ii in range(0, len(images), 50):
            # print(ii, len(images))

            image_sub_list = images[ii:ii + 50]
            inputs = processor(text=[prompt] * len(image_sub_list), images=image_sub_list, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"].cuda(),
                    input_ids=inputs["input_ids"][:, :-1].cuda(),
                    attention_mask=inputs["attention_mask"][:, :-1].cuda(),
                    img_features=None,
                    img_attn_mask=inputs["img_attn_mask"][:, :-1].cuda(),
                    use_cache=True,
                    max_new_tokens=200,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for jj in range(len(image_sub_list)):
                _, entities = processor.post_process_generation(generated_text[jj])
                image_name = img_path[ii:ii + 50][jj].split('/')[-1].split('.')[0]
                image_h = image_sub_list[jj].height
                image_w = image_sub_list[jj].width
                kk = 0
                caption_box_info_single_scene[image_name] = {'boxes': [], 'classes': []}
                for entity_name, (start, end), bboxes in entities:
                    for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
                        orig_x1, orig_y1, orig_x2, orig_y2 = \
                            int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
                        captions_dict_single_scene[f'{image_name}_{kk}'] = entity_name
                        caption_box_info_single_scene[image_name]['boxes'].append([orig_y1, orig_x1, orig_y2, orig_x2])
                        caption_box_info_single_scene[image_name]['classes'].append(entity_name)
                        kk += 1
                caption_box_info_single_scene[image_name]['boxes'] = np.array(caption_box_info_single_scene[image_name]['boxes'])
                caption_box_info_single_scene[image_name]['classes'] = np.array(caption_box_info_single_scene[image_name]['classes'])
        return captions_dict_single_scene, caption_box_info_single_scene


class NuScenesProcessor(ProcessorTemplate):
    def __init__(self, device):
        super(NuScenesProcessor, self).__init__(device)

        from nuscenes.nuscenes import NuScenes
        args.dataset_path = os.path.join(args.dataset_path, args.version)
        self.data_inst = NuScenes(
            version=args.version, dataroot=os.path.join('data/nuscenes', args.version), verbose=True
        )

        # set oss client
        self.oss_client = common_utils.OSSClient() if 's3://' in args.dataset_path else None

        # load info
        info_path = os.path.join(args.dataset_path, args.info_path)
        self.infos = pickle.load(self.oss_client.get(info_path)) if self.oss_client else pickle.load(open(info_path, 'rb'))

        # TODO: enable multiple parts
        if args.all > 1:
            raise NotImplementedError

        if args.split_id != -1 and args.split_total != -1:
            step = [x for x in range(0, len(self.infos), int(len(self.infos) / args.split_total) + 1)]
            step += [len(self.infos)]
            start_idx = max(0, step[args.split_id])
            end_idx = min(len(self.infos), step[args.split_id + 1])
            self.infos = self.infos[start_idx:end_idx]

        print(f'Total {len(self.infos)} samples are loaded.')

    def get_image_from_cam_channel(self, sample_record, cam_channel):
        camera_token = sample_record['data'][cam_channel]
        cam = self.data_inst.get('sample_data', camera_token)
        image_path = os.path.join(args.dataset_path, cam['filename'])
        if self.oss_client:
            image = Image.open(self.oss_client.get(image_path))
        else:
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = np.array(image)
        return image

    def process_scene_caption(self):
        print('Processing scene captions.....')

        # load view caption
        caption_view_path = os.path.join(args.output_dir, 'caption_view_{}_{}_{}_{}.json'.format(
                args.dataset, args.version, args.caption_model.split('/')[-1], args.tag))
        captions_view = json.load(open(caption_view_path, 'r'))
        print(f'load view captions from {caption_view_path}')

        captions_scene = {}

        for frame_name in tqdm(list(captions_view.keys()), total=len(captions_view)):
            text = '. '.join(captions_view[frame_name].values())
            if len(text.split(' ')) > 75:
                sum_caption = self.summarizer(text, max_length=75)[0]['summary_text']
            else:
                sum_caption = text
            captions_scene[frame_name] = sum_caption

        write_caption_to_file(
            captions_scene,
            os.path.join(args.output_dir, 'caption_scene_{}_{}_{}_{}.json'.format(
                args.dataset, args.version, args.caption_model.split('/')[-1], args.tag))
        )

    def process_view_caption(self):
        captions_view = {}

        print('Processing view captions.....')
        for idx, info in tqdm(enumerate(self.infos), total=len(self.infos)):
            sample_token = info['token']
            sample_record = self.data_inst.get('sample', sample_token)
            frame_name = info['lidar_path'].split('.')[0].replace('/', '_')
            image_path = []
            for cam_channel in args.image_list:
                camera_token = sample_record['data'][cam_channel]
                cam = self.data_inst.get('sample_data', camera_token)
                image_path.append(os.path.join(self.data_inst.dataroot, cam['filename']))

            res = self.model.predict_step(image_path, image_name_list=args.image_list)
            captions_view[frame_name] = res

        write_caption_to_file(
            captions_view,
            os.path.join(args.output_dir, 'caption_view_{}_{}_{}_{}.json'.format(
                args.dataset, args.version, args.caption_model.split('/')[-1], args.tag))
        )

    @staticmethod
    def process_detic_crop_caption_with_raw_pred():
        captions_crop = {}
        detic_infos = pickle.load(open(args.detic_info, 'rb'))

        print(f'Processing crop captions with raw prediction from {args.detic_info}')
        # for each scene
        for idx, info in tqdm(enumerate(detic_infos), total=len(detic_infos)):
            scene_name = info['scene_name']
            image_infos = info['infos']

            # for each image
            scene_crop_caption = {}
            for image_name, image_info in image_infos.items():
                image_classes = image_info['classes']
                counter = 0
                for class_name in image_classes:
                    class_name = class_name.replace('_', ' ')
                    scene_crop_caption[f'{image_name.lower()}_{counter}'] = class_name
                    counter += 1
            captions_crop[scene_name] = scene_crop_caption

        write_caption_to_file(
            captions_crop,
            os.path.join(args.output_dir, 'caption_crop_{}_{}_{}_{}.json'.format(
                args.dataset, args.version, args.caption_model.split('/')[-1], args.tag))
        )

    def process_basic_crop_caption_with_caption_idx_info(self):
        caption_basic_crop = {}
        # caption_idx_info = pickle.load(open(args.caption_idx_info, 'rb'))
        window_size = np.array(args.window_size)
        strides = (int(window_size[0] * (1 - args.overlap_ratio)), int(window_size[1] * (1 - args.overlap_ratio)))
        if args.scene_name_split != 'None':
            scene_name_list = [x.strip() for x in open(args.scene_name_split, 'r').readlines()]

        print('Processing basic crop captions.....')
        for idx, info in tqdm(enumerate(self.infos), total=len(self.infos)):
            sample_token = info['token']
            sample_record = self.data_inst.get('sample', sample_token)
            frame_name = info['lidar_path'].split('.')[0].replace('/', '_')
            if args.scene_name_split != 'None':
                if frame_name not in scene_name_list:
                    continue

            caption_basic_crop[frame_name] = {}
            # scene_caption_idx_info = caption_idx_info[idx]
            for cam_channel in tqdm(args.image_list):
                image = self.get_image_from_cam_channel(sample_record, cam_channel)

                # boxes = scene_caption_idx_info['boxes'][cam_channel.lower()]
                boxes = get_sliding_windows(image.shape, window_size, strides)
                cropped_images = self.crop_image_with_boxes(image, boxes)
                num_cropped_image = len(cropped_images)
                cropped_image_name_list = [f'{cam_channel.lower()}_{i}' for i in range(num_cropped_image)]
                res = self.model.predict_step_with_image(cropped_images, image_name_list=cropped_image_name_list)
                caption_basic_crop[frame_name].update(res)

                # plt.imsave('vis_output/origin_img.png', image)
                # import ipdb; ipdb.set_trace(context=20)

        write_caption_to_file(
            caption_basic_crop,
            os.path.join(args.output_dir, 'caption_slidwind_{}_{}_{}_{}.json'.format(
                args.dataset, args.version, args.caption_model.split('/')[-1], args.tag))
        )

    def process_detic_crop_caption_with_caption(self):
        captions_detic_crop_cap = {}
        detic_infos = pickle.load(open(args.detic_info, 'rb'))

        print(f'Processing detic boxes with image captions with from {args.detic_info}')
        # for each scene
        for idx, info in tqdm(enumerate(self.infos), total=len(self.infos)):
            sample_token = info['token']
            sample_record = self.data_inst.get('sample', sample_token)
            scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
            captions_detic_crop_cap[scene_name] = {}

            # boxes prediction from detic info
            assert detic_infos[idx]['scene_name'] == scene_name
            scene_detic_info = detic_infos[idx]['infos']

            for image_name in args.image_list:
                image = self.get_image_from_cam_channel(sample_record, image_name)

                image_detic_pred = scene_detic_info[image_name]
                boxes = image_detic_pred['boxes']
                if boxes.shape[0] == 0:
                    continue
                # enlarge boxes
                if args.enlarge_box_ratio > 1.0:
                    boxes = enlarge_boxes_size(boxes, args.enlarge_box_ratio, args.enlarge_boxes_max_thresh, image.shape)

                cropped_images = self.crop_image_with_boxes(
                    image, boxes, min_image_area=args.enlarge_box_ratio * args.min_image_crop_area
                )
                num_cropped_image = len(cropped_images)
                if num_cropped_image == 0:
                    continue
                cropped_image_name_list = [f'{image_name.lower()}_{i}' for i in range(num_cropped_image)]
                res = self.model.predict_step_with_image(cropped_images, image_name_list=cropped_image_name_list)
                captions_detic_crop_cap[scene_name].update(res)

        caption_path = 'caption_detic_crop_cap_{}_{}_{}_{}.json'.format(
                args.dataset, args.version, args.caption_model.split('/')[-1], args.tag
        )

        write_caption_to_file(captions_detic_crop_cap, os.path.join(args.output_dir, caption_path))

    def process_dense_caption(self):
        model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
        model.cuda()
        processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)

        prompt = "<grounding> Describe this image in detail:"

        print('Processing dense captions.....')
        captions_dict = {}
        caption_box_info = []
        for idx, info in tqdm(enumerate(self.infos), total=len(self.infos)):
            frame_name = info['lidar_path'].split('.')[0].replace('/', '_')
            captions_dict[frame_name], caption_box_info_single_scene = \
                self.process_dense_caption_single_scene(info, model, processor, prompt)
            caption_box_info.append(
                {'infos': caption_box_info_single_scene, 'scene_name': frame_name})

        caption_path = os.path.join(args.output_dir, 'caption_dense_{}_{}_{}.json'.format(
                args.dataset, args.version, args.tag))
        write_caption_to_file(captions_dict, caption_path)

        caption_box_info_path = os.path.join(args.output_dir, 'caption_dense_box_info_{}_{}_{}.pkl'.format(
            args.dataset, args.version, args.tag))
        if args.split_id != -1 and args.split_total != -1:
            caption_box_info_path = caption_box_info_path[:-4] + f'_part{args.split_id}' + caption_box_info_path[-4:]
        with open(caption_box_info_path, 'wb') as f:
            pickle.dump(caption_box_info, f)

    def process_dense_caption_single_scene(self, info, model, processor, prompt):
        sample_token = info['token']
        sample_record = self.data_inst.get('sample', sample_token)
        # frame_name = info['lidar_path'].split('.')[0].replace('/', '_')
        img_path = []

        for cam_channel in args.image_list:
            camera_token = sample_record['data'][cam_channel]
            cam = self.data_inst.get('sample_data', camera_token)
            img_path.append((os.path.join(self.data_inst.dataroot, cam['filename']), cam_channel))

        captions_dict_single_scene = {}
        caption_box_info_single_scene = {}
        images = []
        for img_ in img_path:
            # image_name = img_.split('/')[-1].split('.')[0]
            image = Image.open(img_[0])
            images.append(image)

        for ii in range(0, len(images), 50):
            # print(ii, len(images))

            image_sub_list = images[ii:ii + 50]
            inputs = processor(text=[prompt] * len(image_sub_list), images=image_sub_list, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"].cuda(),
                    input_ids=inputs["input_ids"][:, :-1].cuda(),
                    attention_mask=inputs["attention_mask"][:, :-1].cuda(),
                    img_features=None,
                    img_attn_mask=inputs["img_attn_mask"][:, :-1].cuda(),
                    use_cache=True,
                    max_new_tokens=200,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for jj in range(len(image_sub_list)):
                _, entities = processor.post_process_generation(generated_text[jj])
                image_name = img_path[ii:ii + 50][jj][1].upper()
                image_h = image_sub_list[jj].height
                image_w = image_sub_list[jj].width
                kk = 0
                caption_box_info_single_scene[image_name] = {'boxes': [], 'classes': []}
                for entity_name, (start, end), bboxes in entities:
                    for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
                        orig_x1, orig_y1, orig_x2, orig_y2 = \
                            int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
                        captions_dict_single_scene[f'{image_name}_{kk}'] = entity_name
                        caption_box_info_single_scene[image_name]['boxes'].append([orig_y1, orig_x1, orig_y2, orig_x2])
                        caption_box_info_single_scene[image_name]['classes'].append(entity_name)
                        kk += 1
                caption_box_info_single_scene[image_name]['boxes'] = np.array(caption_box_info_single_scene[image_name]['boxes'])
                caption_box_info_single_scene[image_name]['classes'] = np.array(caption_box_info_single_scene[image_name]['classes'])
        return captions_dict_single_scene, caption_box_info_single_scene


class KittiProcessor(ProcessorTemplate):
    def __init__(self, device):
        super(KittiProcessor, self).__init__(device)

        args.info_path = os.path.join(args.dataset_path, 'img_shape_dict.pkl')
        with open(args.info_path, 'rb') as fin:
            self.infos = list(pickle.load(fin).keys())

        # set oss client
        self.oss_client = common_utils.OSSClient() if 's3://' in args.dataset_path else None

        # TODO: enable multiple parts
        if args.all > 1:
            raise NotImplementedError
        if args.split_id != -1 and args.split_total != -1:
            step = [x for x in range(0, len(self.infos), int(len(self.infos) / args.split_total) + 1)]
            step = step + [len(self.infos)]
            start_idx = max(0, step[args.split_id])
            end_idx = min(len(self.infos), step[args.split_id + 1])
            self.infos = self.infos[start_idx:end_idx]

        print(f'Total {len(self.infos)} samples are loaded.')

    def get_image_from_cam_channel(self, image_path):
        if self.oss_client:
            image = Image.open(self.oss_client.get(image_path))
        else:
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = np.array(image)
        return image

    def process_view_caption(self):
        captions_view = {}

        print('Processing view captions.....')
        image_list = []
        for idx, info in tqdm(enumerate(self.infos), total=len(self.infos)):

            image_path = os.path.join(args.dataset_path, 'dataset', '/'.join(info.split('_'))).replace('image/2', 'image_2') + '.png'
            # image_list.append(image_path)

            res = self.model.predict_step([image_path], image_name_list=['0'])
            captions_view[info] = {}
            captions_view[info] = res

        write_caption_to_file(
            captions_view,
            os.path.join(args.output_dir, 'caption_view_{}_{}_{}_{}.json'.format(
                args.dataset, args.caption_model.split('/')[-1], args.tag))
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('language model')
    parser.add_argument('--caption_model', default='nlpconnect/vit-gpt2-image-captioning',
                        choices=['nlpconnect/vit-gpt2-image-captioning',
                                 'damo/ofa_image-caption_coco_large_en',
                                 'blip2_t5'],
                        type=str, help='language model name')
    parser.add_argument('--summarizer', default='facebook/bart-large-cnn', type=str, help='language model name')
    parser.add_argument('--dataset', default='scannet', type=str, help='dataset name')
    parser.add_argument('--output_dir', required=True, help='path to output folder')
    parser.add_argument('--max_length', default=64, type=int, help='max length')
    parser.add_argument('--caption_mode', default='view_caption',
                        choices=['view_caption', 'scene_caption', 'detic_raw_pred',
                                 'basic_crop_caption', 'detic_crop_caption', 'kosmos2_dense_caption'])
    parser.add_argument('--dataset_path', default='../data/scannetv2', type=str, help='language model name')
    parser.add_argument('--tag', default='', type=str, help='')
    parser.add_argument('--ofa_ckpt_dir', default='', type=str, help='')
    parser.add_argument('--scene_name_split', default='None', type=str,
                        help='specify scene names need to generate captions')

    # for speed up
    parser.add_argument('--cur', default=1, type=int, help='')
    parser.add_argument('--all', default=1, type=int, help='')

    # scannet args
    parser.add_argument('--image_tag', default='scannet_frames_25k', type=str, help='')

    # nuscenes args
    parser.add_argument('--info_path', default='nuscenes_infos_1sweeps_train.pkl', type=str, help='language model name')
    parser.add_argument('--version', default='v1.0-trainval', type=str, help='')
    parser.add_argument('--image_list', default=[
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'], type=list, help='')
    # for detic crop caption only
    parser.add_argument('--detic_info', default='nuscenes_v1.0-mini_detic_pred_results.pkl', type=str, help='')

    # for basic crop caption
    parser.add_argument('--caption_idx_info', default='nuscenes_caption_idx_basic_crop.pkl', type=str, help='')
    parser.add_argument('--window_size', default=(400, 500), type=tuple, help='window size for cropping sub images')
    parser.add_argument('--overlap_ratio', default=0.3, type=float, help='overlap ratio when crop images')

    # detic crop caption
    parser.add_argument('--min_image_crop_area', default=3000, type=int, help='')
    parser.add_argument('--enlarge_boxes_max_thresh', default=8000, type=int, help='maximum size that dont need a enlarge')
    parser.add_argument('--enlarge_box_ratio', default=1.0, type=float, help='enlarge the box with a ratio')
    parser.add_argument('--use_detic_raw_pred_as_suppl', action='store_true', default=False, help='')

    # To split into different
    parser.add_argument('--split_id', default=-1, type=int, help='start from 0')
    parser.add_argument('--split_total', default=-1, type=int, help='')
    global args
    args = parser.parse_args()

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'scannet':
        processor = ScanNetProcessor(device)
    elif args.dataset == 'nuscenes':
        processor = NuScenesProcessor(device)
    elif args.dataset == 'kitti':
        processor = KittiProcessor(device)
    else:
        raise NotImplementedError

    if args.caption_mode == 'view_caption':
        processor.process_view_caption()
    elif args.caption_mode == 'scene_caption':
        processor.process_scene_caption()
    elif args.caption_mode == 'detic_raw_pred':
        """
        python -m tools.process_tools.generate_caption --caption_mode detic_raw_pred --dataset_path data/nuscenes \
        --dataset nuscenes --version v1.0-mini --output_dir data/nuscenes/text_embed \
        --detic_info ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl  \
        --tag detic_raw_pred
        """
        processor.process_detic_crop_caption_with_raw_pred()
    elif args.caption_mode == 'basic_crop_caption':
        """
        python -m tools.process_tools.generate_caption --caption_model damo/ofa_image-caption_coco_large_en \
         --caption_mode basic_crop_caption --dataset_path data/nuscenes --dataset nuscenes
         --version v1.0-mini --output_dir data/nuscenes/text_embed \
        --tag w400-500_overlap0.3 --ofa_ckpt_dir ../OFA-large-caption/
        """
        processor.process_basic_crop_caption_with_caption_idx_info()
    elif args.caption_mode == 'detic_crop_caption':
        """
        python tools/process_tools/generate_caption.py --caption_mode detic_crop_caption --dataset_path data/nuscenes \
        --dataset nuscenes --version v1.0-mini --output_dir data/nuscenes/text_embed \
        --detic_info ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl
        ############################
        #  with enlarge box crop ###
        ############################
        python tools/process_tools/generate_caption.py --caption_mode detic_crop_caption --dataset_path data/nuscenes \
        --dataset nuscenes --version v1.0-mini --output_dir data/nuscenes/text_embed \
        --detic_info ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl \
        --tag enlarge2.5 --enlarge_box_ratio 2.5
        """
        processor.process_detic_crop_caption_with_caption()
    elif args.caption_mode == 'kosmos2_dense_caption':
        """
        python tools/process_tools/generate_caption.py --caption_mode kosmos2_dense_caption --dataset_path data/scannet \
        --dataset scannet --version '' --output_dir data/scannet/text_embed \
        --tag kosmos2_densecap --image_tag scannet_images_125k_1296
        """
        processor.process_dense_caption()
    else:
        raise NotImplementedError

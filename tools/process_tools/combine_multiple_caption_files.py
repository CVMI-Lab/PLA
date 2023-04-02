import json
import argparse
import tqdm
import pickle


def write_caption_to_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'The caption is dump to {path}')


def replace_dict_keys_with_new_keys(origin_dict, new_key_list):
    curr_key_list = list(origin_dict.keys())
    new_dict = {}
    for i, key in enumerate(curr_key_list):
        new_dict[new_key_list[i]] = origin_dict[key]

    return new_dict


def merge_captions_with_path_list(caption_path_list, caption_save_path):
    new_caption = {}
    scene_caption_num = {}

    for caption_path in caption_path_list:
        current_caption = json.load(open(caption_path, 'r'))
        for scene_name, curr_scene_caption in tqdm.tqdm(current_caption.items(), total=len(current_caption)):
            counter = scene_caption_num[scene_name] if scene_name in scene_caption_num else 0

            image_name_list = [f'{counter + i}' for i in range(len(curr_scene_caption))]
            new_scene_caption = replace_dict_keys_with_new_keys(curr_scene_caption, image_name_list)
            if scene_name in new_caption:
                new_caption[scene_name].update(new_scene_caption)
            else:
                new_caption[scene_name] = new_scene_caption

            counter += len(curr_scene_caption)
            scene_caption_num[scene_name] = counter

    write_caption_to_file(new_caption, caption_save_path)


def merge_caption_idx_with_path_list(caption_idx_path_list, caption_idx_save_path):
    new_caption_idx = []
    caption_idx_list = []
    for caption_idx_path in caption_idx_path_list:
        caption_idx = pickle.load(open(caption_idx_path, 'rb'))
        caption_idx_list.append(caption_idx)

    for i in tqdm.tqdm(range(len(caption_idx_list[0]))):
        scene_caption = {}
        scene_caption_infos = {}
        counter = 0
        for _, caption_idx in enumerate(caption_idx_list):
            if 'scene_name' not in scene_caption:
                scene_caption['scene_name'] = caption_idx[i]['scene_name']

            new_image_name_list = [f'{counter + i}' for i in range(len(caption_idx[i]['infos']))]
            new_scene_caption_idx = replace_dict_keys_with_new_keys(caption_idx[i]['infos'], new_image_name_list)

            scene_caption_infos.update(new_scene_caption_idx)
            counter += len(caption_idx[i]['infos'])

        scene_caption['infos'] = scene_caption_infos
        new_caption_idx.append(scene_caption)

    with open(caption_idx_save_path, 'wb') as f:
        pickle.dump(new_caption_idx, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--caption_path_list',
                        default=['data/nuscenes/text_embed/caption_basic_crop_nuscenes_v1.0-mini_vit-gpt2-image'
                                 '-captioning_w400-500_overlap0.3.json',
                                 'data/nuscenes/text_embed/caption_detic_crop_cap_nuscenes_v1.0-mini_vit-gpt2-image'
                                 '-captioning_.json'],
                        type=list, help='')
    parser.add_argument('--caption_idx_path_list',
                        default=['data/nuscenes/v1.0-mini/nuscenes_caption_idx_basic_crop.pkl',
                                 'data/nuscenes/v1.0-mini/nuscenes_caption_idx_detic_crop_cap.pkl'],
                        type=list, help='')
    parser.add_argument('--caption_save_path', required=True, type=str, help='')
    parser.add_argument('--caption_idx_save_path', required=True, type=str, help='')

    args = parser.parse_args()

    print('Start to merge captions ........')
    merge_captions_with_path_list(args.caption_path_list, args.caption_save_path)
    print('Finish merging captions ........')

    print('Start to merge captions idx file ........')
    merge_caption_idx_with_path_list(args.caption_idx_path_list, args.caption_idx_save_path)
    print('Finish merging captions idx file ........')


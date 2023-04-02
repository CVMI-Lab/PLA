import argparse
import pickle
import json
import tqdm


def write_caption_to_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'The caption is dump to {path}')


def filter_captions_without_points(caption_info, caption_idx_info):
    for idx, scene_caption_idx_info in tqdm.tqdm(enumerate(caption_idx_info), total=len(caption_idx_info)):
        scene_name = scene_caption_idx_info['scene_name']
        scene_caption_idx = scene_caption_idx_info['infos']
        scene_captions = caption_info[scene_name]

        image_name_list = list(scene_caption_idx.keys())
        for image_name in image_name_list:
            image_caption_idx = scene_caption_idx[image_name]
            if image_caption_idx.shape[0] == 0:
                scene_caption_idx.pop(image_name)
                scene_captions.pop(image_name)

    write_caption_to_file(caption_info, args.save_caption_info_path)
    with open(args.save_caption_idx_info_path, 'wb') as f:
        pickle.dump(caption_idx_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--caption_info_path', type=str, help='')
    parser.add_argument('--caption_idx_info_path', type=str, help='')

    parser.add_argument('--save_caption_info_path', type=str, help='')
    parser.add_argument('--save_caption_idx_info_path', type=str, help='')

    global args
    args = parser.parse_args()

    caption_info = json.load(open(args.caption_info_path, 'r'))
    caption_idx_info = pickle.load(open(args.caption_idx_info_path, 'rb'))

    filter_captions_without_points(caption_info, caption_idx_info)





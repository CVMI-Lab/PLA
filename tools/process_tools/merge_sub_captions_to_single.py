import json
import argparse
import glob
import os
import tqdm


def write_caption_to_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'The caption is dump to {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--caption_prefix', required=True, type=str, help='')
    parser.add_argument('--caption_dir', required=True, type=str, help='')
    parser.add_argument('--caption_save_path', required=True, type=str, help='')
    args = parser.parse_args()

    sub_caption_path_list = glob.glob(os.path.join(args.caption_dir, args.caption_prefix + '*.json'))

    print(sub_caption_path_list)

    whole_caption = {}
    total_num = 0
    for sub_caption_path in tqdm.tqdm(sub_caption_path_list):
        sub_caption = json.load(open(sub_caption_path, 'r'))
        whole_caption.update(sub_caption)
        total_num += len(sub_caption)

    write_caption_to_file(whole_caption, args.caption_save_path)
    print(f'Total {total_num} captions are merged.')

import os
import json
import copy
import random
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
# ks50
parser.add_argument('--clean-path', type=str, default='code_path/json_csv_files/ks50/clean/severity_0.json')
parser.add_argument('--video-c-path', type=str, default="data_path/Kinetics50/image_mulframe_val256_k=50-C")
parser.add_argument('--audio-c-path', type=str, default="data_path/Kinetics50/audio_val256_k=50-C")
parser.add_argument('--dataset', type=str, default='ks50', choices=['vggsound', 'ks50'])
# vgg
# parser.add_argument('--clean-path', type=str, default='code_path/json_csv_files/vgg/clean/severity_0.json')
# parser.add_argument('--video-c-path', type=str, default="data_path/VGGSound/image_mulframe_test-C")
# parser.add_argument('--audio-c-path', type=str, default="data_path/VGGSound/audio_test-C")
# parser.add_argument('--dataset', type=str, default='vggsound', choices=['vggsound', 'ks50'])
args = parser.parse_args()

with open(args.clean_path, 'r') as f:
    data = json.load(f)

tmp_dic_list = data['data']

corruption_list_a = ['gaussian_noise', 'crowd', 'rain', 'thunder', 'traffic', 'wind']
corruption_list_v = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                     'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
                     'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
if args.dataset == 'ks50':
    for corruption_v in corruption_list_v:
        for corruption_a in corruption_list_a:
            save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'both')

            if not os.path.exists(os.path.join(save_path, corruption_v)):
                os.makedirs(os.path.join(save_path, corruption_v))
            dic_list = []
            for dic in tmp_dic_list:
                new_dic = {
                    "video_id": dic.get("video_id"), # + '-{}-{}'.format(method, severity),
                    "wav": os.path.join(args.audio_c_path, corruption_a, 'severity_5', '{}.wav'.format(dic.get("video_id"))),
                    "video_path": os.path.join(args.video_c_path, '{}/severity_5/'.format(corruption_v)),
                    "labels": dic.get("labels")
                }
                dic_list.append(new_dic)
            print(len(dic_list))
            random.shuffle(dic_list)
            new_json = {"data": dic_list}
            with open(os.path.join(save_path, corruption_v, '{}_severity_5.json'.format(corruption_a)), "w") as file1:
                json.dump(new_json, file1, indent=1)
elif args.dataset == 'vgg':
    for corruption_a in corruption_list_a:
        for corruption_v in corruption_list_v:
            save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'both')

            if not os.path.exists(os.path.join(save_path, corruption_a)):
                os.makedirs(os.path.join(save_path, corruption_a))
            dic_list = []
            for dic in tmp_dic_list:
                new_dic = {
                    "video_id": dic.get("video_id"),  # + '-{}-{}'.format(method, severity),
                    "wav": os.path.join(args.audio_c_path, corruption_a, 'severity_5', '{}.wav'.format(dic.get("video_id"))),
                    "video_path": os.path.join(args.video_c_path, '{}/severity_5/'.format(corruption_v)),
                    "labels": dic.get("labels")
                }
                dic_list.append(new_dic)
            print(len(dic_list))
            random.shuffle(dic_list)
            new_json = {"data": dic_list}
            with open(os.path.join(save_path, corruption_a, '{}_severity_5.json'.format(corruption_v)), "w") as file1:
                json.dump(new_json, file1, indent=1)

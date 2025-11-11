import os
import json
import random
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
# ks50
parser.add_argument('--clean-path', type=str, default='code_path/json_csv_files/ks50/clean/severity_0.json')
parser.add_argument('--video-c-path', type=str, default="data_path/Kinetics50/image_mulframe_val256_k=50-C")
parser.add_argument('--audio-path', type=str, default="data_path/Kinetics50/audio_val256_k=50")
# vgg
# parser.add_argument('--clean-path', type=str, default='code_path/json_csv_files/vgg/clean/severity_0.json')
# parser.add_argument('--video-c-path', type=str, default="data_path/VGGSound/image_mulframe_test-C")
# parser.add_argument('--audio-path', type=str, default="data_path/VGGSound/audio_test")
parser.add_argument('--corruption', nargs='*', default=['all'])
args = parser.parse_args()

with open(args.clean_path, 'r') as f:
    data = json.load(f)

tmp_dic_list = data['data']

severity_list = range(1, 6)
if args.corruption[0] == 'all':
    corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',

    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',

    'snow',
    'frost',
    'fog',
    'brightness',

    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    ]
else:
    corruption_list = args.corruption

for corruption in corruption_list:
    for severity in severity_list:
        save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'video')

        if not os.path.exists(os.path.join(save_path, corruption)):
            os.makedirs(os.path.join(save_path, corruption))
        dic_list = []
        for dic in tmp_dic_list:
            new_dic = {
                "video_id": dic.get("video_id"), # + '-{}-{}'.format(method, severity),
                "wav": os.path.join(args.audio_path, '{}.wav'.format(dic.get("video_id"))),
                "video_path": os.path.join(args.video_c_path, '{}/severity_{}/'.format(corruption, severity)),
                "labels": dic.get("labels")
            }
            dic_list.append(new_dic)
        print(len(dic_list))
        random.shuffle(dic_list)
        new_json = {"data": dic_list}
        with open(os.path.join(save_path, corruption, 'severity_{}.json'.format(severity)), "w") as file1:
            json.dump(new_json, file1, indent=1)

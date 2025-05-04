import pickle
import os
import numpy as np
import shutil
import argparse
import cv2
import json

def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='block_hammer_beat',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    head_camera_type = args.head_camera_type
    load_dir = f'data/{task_name}_{head_camera_type}_pkl'

    save_dir = f'./policy/Diffusion-Policy/data/{task_name}_{head_camera_type}_{num}_png/validation'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    meta_data = {}

    current_ep = 90

    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        episode_dir = os.path.join(save_dir, f'episode{current_ep}')
        os.makedirs(episode_dir)
        file_num = 0
        num_frames = 0

        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)

            head_img = data['observation']['head_camera']['rgb']
            left_img = data['observation']['left_camera']['rgb']
            right_img = data['observation']['right_camera']['rgb']
            front_img = data['observation']['front_camera']['rgb']

            # 保存图片，使用 BGR 格式
            cv2.imwrite(os.path.join(episode_dir, f'{file_num}_head.png'), cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(episode_dir, f'{file_num}_left.png'), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(episode_dir, f'{file_num}_right.png'), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(episode_dir, f'{file_num}_front.png'), cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))

            file_num += 1
            num_frames += 1

        current_ep += 1

        meta_data[str(current_ep - 1)] = {
            "text": "stack the blocks",
            "num_frames": num_frames
        }

    # 保存 meta.json 文件
    with open(os.path.join(save_dir, 'meta.json'), 'w') as meta_file:
        json.dump(meta_data, meta_file, indent=4)

    print(f'\nProcessing complete. Data saved to {save_dir}')

if __name__ == '__main__':
    main()
import pickle
import os
import numpy as np
import shutil
import argparse
import cv2

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

    save_dir = f'./calvin_Robotwin/{task_name}/validation'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    lang_annotations_dir = os.path.join(save_dir, 'lang_annotations')
    os.makedirs(lang_annotations_dir)

    info_indx = []
    language_task = []
    language_ann = []

    current_ep = 0

    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        episode_dir = os.path.join(save_dir, f'episode{current_ep}')
        os.makedirs(episode_dir)
        file_num = 0
        num_frames = 0
        start_frame = 0

        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)

            head_img = data['observation']['head_camera']['rgb']
            # right_img = data['observation']['right_camera']['rgb']
            front_img = data['observation']['front_camera']['rgb']
            robot_obs = data['joint_action']
            # endpose = data['endpose']
            # print('joint_action:', robot_obs)

            # 保存 .npz 文件
            npz_filename = os.path.join(episode_dir, f'frame_{file_num:04d}.npz')
            np.savez_compressed(npz_filename, robot_obs=robot_obs, rgb_static=head_img, rgb_gripper=front_img)

            file_num += 1
            num_frames += 1

        end_frame = start_frame + num_frames - 1
        info_indx.append((start_frame, end_frame))
        language_task.append(f'{task_name}')
        language_ann.append('stack the blocks')
        start_frame = end_frame + 1

        current_ep += 1

    # 保存 auto_lang_ann.npy 文件
    auto_lang_ann = {
        'info': {
            'indx': info_indx
        },
        'language': {
            'task': language_task,
            'ann': language_ann
        }
    }
    np.save(os.path.join(lang_annotations_dir, 'auto_lang_ann.npy'), auto_lang_ann)

    print(f'\nProcessing complete. Data saved to {save_dir}')

if __name__ == '__main__':
    main()
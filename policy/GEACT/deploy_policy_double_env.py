import os
import sys
import numpy as np

# 暴露给 policy_model_server.py 的 get_model
# from .act_model import get_model as _get_model
try:
    from .act_model import _get_model
except:
    pass


def encode_obs(observation):
    # 从 RoboTwin env 的观测抽取三视角和关节状态
    img_front = observation["observation"]["head_camera"]["rgb"]
    img_right = observation["observation"]["right_camera"]["rgb"]
    img_left  = observation["observation"]["left_camera"]["rgb"]
    input_rgb_arr = [img_front, img_right, img_left]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    return _get_model(usr_args)


def eval(TASK_ENV, model, observation):
    # model 为 ModelClient（客户端），在服务端则是 GEActModel 实例
    # 初始化语言一次（按需）
    if model.call(func_name='get_observation_window') is None:
        instruction = TASK_ENV.get_instruction()
        model.call(func_name='set_language', obs=instruction)

    # 首次更新观察
    img_arr, state = encode_obs(observation)
    model.call(func_name='update_observation_window', obs=(img_arr, state))

    # ###debug
    # obs=     model.call(func_name='get_observation_window')
    # print(f"Current observation window: {obs}")
    # 获取动作（返回未来若干步）
    actions = model.call(func_name='get_action')
    # 如果需要限制步数（与 pi0 一致）
    # actions = actions[:model.call(func_name='get_pi0_step')]

    for action in actions:
        TASK_ENV.take_action(action)
        # 每步刷新观察窗口
        observation = TASK_ENV.get_obs()
        img_arr, state = encode_obs(observation)
        model.call(func_name='update_observation_window', obs=(img_arr, state))


def reset_model(model):
    model.call(func_name='reset_model')
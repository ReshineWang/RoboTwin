import os
import numpy as np
import torch
import argparse
import traceback


# 让GEAct模块可导入
GE_ROOT = os.path.abspath(os.path.dirname(__file__))
GE_UTILS = os.path.join(GE_ROOT, "web_infer_utils")
import sys
sys.path.insert(0, GE_UTILS)
sys.path.insert(0, GE_ROOT)

from web_infer_utils.MVActor import MVActor


class GEActModel:
    def __init__(self, config_file, transformer_file, domain_name="RoboTwin",
                 add_state=True, threshold=None, n_prev =4, denoise_step=5,
                 action_dim=14, gripper_dim=1, execution_step=7):
        self.actor = MVActor(
            config_file=config_file,
            transformer_file=transformer_file,
            threshold=threshold,
            n_prev=n_prev,
            action_dim=action_dim,
            gripper_dim=gripper_dim,
            domain_name=domain_name,
            load_weights=True,
            num_inference_steps=denoise_step,
            device = torch.device("cuda:0"),
            dtype = torch.bfloat16,
        )
        self.execution_step = execution_step
        self.add_state = add_state
        self.instruction = ""
        self._img_arr = None
        self._state = None

    # 与RPC对齐的方法
    def set_language(self, instruction: str):
        self.instruction = instruction
        return True

    def update_observation_window(self, img_arr, state):
        # img_arr: list of 3 np.uint8 images (H,W,3)
        # state: np.ndarray shape (C,)
        self._img_arr = img_arr
        self._state = state
        return True

    def get_action(self):
        # 将3视角拼为(v,h,w,3)，v=3
        assert self._img_arr is not None, "Call update_observation_window first"
        
        # --- 新增逻辑：检查 resize 并处理图像 ---
        # 获取 resize 参数，默认为 None
        target_size = getattr(self.actor, 'resize', None)
        
        current_imgs = self._img_arr
        if target_size is not None:
            import cv2
            # 兼容 target_size 是整数 (e.g. 224) 或元组 (e.g. (224, 224)) 的情况
            dsize = (target_size, target_size) if isinstance(target_size, int) else target_size
            dsize = (dsize[1], dsize[0])  # 修正顺序，确保 (width, height)

            # 对列表中的每一帧图像进行 resize
            current_imgs = [cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR) for img in current_imgs]
        # -------------------------------------

        obs = np.stack(current_imgs, axis=0)  # (3, H, W, 3), uint8
        
        prompt = self.instruction or ""
        # GEAct 的 play 接口：obs(ndarray), prompt(str), execution_step(int), state(ndarray|None)
        state = self._state if self.add_state else None
        
        print(f"observation shape: {obs.shape}")
        print(f"Playing with prompt: {prompt}, execution_step: {self.execution_step}, state included: {self.add_state}")
        
        actions = self.actor.play(obs=obs, prompt=prompt,
                                  execution_step=self.execution_step,
                                  state=state)
        # 返回 ndarray，服务器编码器已支持
        return actions

    def reset_model(self):
        self.instruction = ""
        self._img_arr = None
        self._state = None
        self.actor.reset()
        return True

    # 如果客户端用到
    def get_observation_window(self):
        if self._img_arr is None:
            return None
        return dict(images=self._img_arr, state=self._state, prompt=self.instruction)

    # 可选：供客户端查询步数
    def get_pi0_step(self):
        return int(self.execution_step)


def _get_model(usr_args):
    # 与 policy_model_server.py 传入的 usr_args 对齐
    # 期待 usr_args 中包含:
    # - train_config_name: GEAct YAML 配置文件路径
    # - model_name: 权重文件路径
    # - ckpt_setting: 可忽略/记录
    # - pi0_step 或 execution_step: 每次返回的未来步数
    # 也可用 overrides 注入：denoise_step, action_dim, add_state, threshold
    config_file = usr_args["config"]
    transformer_file = usr_args["weight"]
    execution_step = usr_args.get("execution_step", 7)
    denoise_step = usr_args.get("denoise_step", 5)
    action_dim = usr_args.get("action_dim", 14)  # 兜底
    add_state = True  # GEAct常用带state，若不需要可以改为 usr_args.get("add_state", True)
    threshold = usr_args.get("threshold", 20)
    domain_name = "RoboTwin"
    return GEActModel(
        config_file=config_file,
        transformer_file=transformer_file,
        domain_name=domain_name,
        add_state=add_state,
        threshold=threshold,
        denoise_step=denoise_step,
        action_dim=action_dim,
        gripper_dim=1,
        execution_step=execution_step,
    )


def _run_debug(args):
    """Quick local debug: load model then run one forward with random obs/state."""
    model = GEActModel(
        config_file=args.config,
        transformer_file=args.weight,
        domain_name=args.domain_name,
        add_state=args.add_state,
        threshold=args.threshold,
        n_prev=args.n_prev,
        denoise_step=args.denoise_step,
        action_dim=args.action_dim,
        gripper_dim=1,
        execution_step=args.execution_step,
    )

    # random obs/state aligned to expected shapes
    obs = np.random.randint(0, 256, size=(3, args.height, args.width, 3), dtype=np.uint8)
    print(f"model.actor.resize: {getattr(model.actor, 'resize', None)}")
    print(f"[DEBUG] obs shape: {obs.shape}")
        # --- 添加 resize 逻辑 ---
    target_size = getattr(model.actor, 'resize', None)
    if target_size is not None:
        import cv2
        # 兼容 target_size 是整数或列表/元组的情况
        dsize = (target_size, target_size) if isinstance(target_size, int) else tuple(target_size)
        dsize = (dsize[1], dsize[0])  # 修正顺序，确保 (width, height)
        obs = np.stack([cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR) for img in obs], axis=0)
        print(f"[DEBUG] obs shape after resize: {obs.shape}")
    # ------------------------


    state = np.random.randn(args.action_dim).astype(np.float32)

    try:
        actions = model.actor.play(
            obs=obs,
            prompt="debug run",
            execution_step=args.execution_step,
            state=state if args.add_state else None,
        )
        print("[DEBUG] actions shape:", getattr(actions, "shape", None))
    except Exception as e:
        print("[DEBUG] Exception during play():", e)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to GEACT config yaml")
    parser.add_argument("--weight", type=str, required=True, help="Path to transformer weight")
    parser.add_argument("--execution_step", type=int, default=30)
    parser.add_argument("--denoise_step", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--n_prev", type=int, default=4)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--add_state", action="store_true", default=True)
    parser.add_argument("--domain_name", type=str, default="RoboTwin")
    args = parser.parse_args()

    _run_debug(args)
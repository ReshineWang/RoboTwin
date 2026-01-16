import os
import time

tasks = [
    # "adjust_bottle", 
    "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave", "pick_diverse_bottles",
    "pick_dual_bottles", "place_a2b_left", "place_a2b_right", "place_bread_basket",
    "place_bread_skillet", "place_burger_fries", "place_can_basket", "place_cans_plasticbox",
    "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_mouse_pad", "place_object_basket", "place_object_scale", "place_object_stand",
    "place_phone_stand", "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object", "shake_bottle_horizontally",
    "shake_bottle", "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
    "stack_bowls_two", "stamp_seal", "turn_switch"
]

def run_task():
    for i, task in enumerate(tasks):
        gpu_id = i % 8  # 假设有8块GPU，循环使用GPU 0-7
        # 会话名称加个前缀，方便管理
        session_name = f"RT_{i}_{task[:10]}" 
        
        # --- 核心改进：使用 conda run 并保留窗口内容 ---
        # 1. cd 进目录
        # 2. conda run -n 环境名 执行后续命令
        # 3. ; read 确保报错后窗口不关闭，方便你 debug
        inner_cmd = (
            f"cd /data/dex/RoboTwin && "
            f"conda run -n RoboTwin --no-capture-output "
            f"bash collect_data.sh {task} demo_clean_vidar {gpu_id}; "
            f"echo '--- 任务结束，按回车关闭窗口 ---'; read"
        )
        
        print(f"[{i+1}/{len(tasks)}] 正在尝试启动: {task} (GPU: {gpu_id})")
        
        # 执行命令
        os.system(f'tmux new-session -d -s {session_name} "{inner_cmd}"')
        
        # 实时检测是否启动成功
        time.sleep(2) # 给 tmux 2秒启动时间
        check = os.popen(f"tmux has-session -t {session_name} 2>/dev/null").read()
        
        # 获取 tmux 运行状态
        if os.system(f"tmux has-session -t {session_name} 2>/dev/null") == 0:
            print(f"  ✅ 启动成功! 窗口名: {session_name}")
        else:
            print(f"  ❌ 启动失败! 请手动检查命令是否正确。")

        if i < len(tasks) - 2:
            print("⏳ 等待 2 分钟后启动下一个...")
            time.sleep(120)

if __name__ == "__main__":
    run_task()
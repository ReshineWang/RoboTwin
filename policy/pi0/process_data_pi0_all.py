import os
import subprocess

# 格式化后的任务列表
tasks = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "c"move_playingcard_away", "move_stapler_pad",lick_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", 
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
    "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
    "rotate_qrcode", "scan_object", "shake_bottle_horizontally", "shake_bottle",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
    "stamp_seal", "turn_switch"
]

def run_sequential_tasks():
    setting = "demo_clean"
    expert_data_num = "50"
    
    total = len(tasks)
    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"正在处理 [{i+1}/{total}]: {task}")
        print(f"{'='*60}")
        
        # 构建命令 (直接调用 python 脚本)
        cmd = f"python scripts/process_data.py {task} {setting} {expert_data_num}"
        
        # 使用 subprocess 运行并实时打印输出
        # 这会等待当前任务完成后再进行下一个循环
        try:
            result = subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 任务 {task} 处理出错，跳过并继续下一个。")
            continue

    print("\n✅ 所有数据处理任务已完成！")

if __name__ == "__main__":
    run_sequential_tasks()
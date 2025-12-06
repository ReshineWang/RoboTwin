from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="demo_clean_repo",
    root="/data/dex/RoboTwin/data/lerobot_data/huggingface/lerobot/111",
)

print(ds)                           # 看一下总帧数和 feature
item = ds[0]
print(item.keys())                  # 里面应该有 observation.images.cam_high 等 key
print(item["observation.images.cam_high"].shape)
import os

mode = os.environ.get("GEACT_MODE", "single").lower()

if mode == "double":
    # 双环境版本：导出 get_model/eval/reset_model 等
    from .deploy_policy_double_env import *
else:
    # 单环境版本（默认）
    from .deploy_policy import *
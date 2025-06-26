# #!/usr/bin/env python3
# import os, sys
# # —— 确保项目根目录在 module 路径中 —— 
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# import pandas as pd
# from preprocess import prepare_pair_df_from_merged
# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
# from envs.env_rl import RL_FixAmt_PairTrade


# # # train_rl1.py 顶部，补充：
# # from gym.wrappers import TimeLimit

# def make_env(df, algo_name):
#     def _init():
#         return RL_FixAmt_PairTrade(
#             df=df,
#             model=algo_name,    # 比如 "A2C" 或 "PPO"
#             tc=0.0002,          # 0.02% 手续费
#             cash=1.0,           # 初始净值 1.0
#             fixed_amt=0.10,      # 每次投入净值的 10%
#             verbose=1           # 打开日志
#         )
#     return _init


# if __name__ == "__main__":
#     # 1️⃣ 读入合并后的原始价格数据
#     #    请确保 data/processed_pair.csv 已经存在，列为 datetime, close0, close1
#     raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])

#     # 2️⃣ 调用预处理：计算 spread、rolling mean/std、zscore
#     df = prepare_pair_df_from_merged(raw, window=900)
#         # —— debug: 打印 df 大小和示例行 —— 
#     print(f"[DEBUG] df shape = {df.shape}")  
#     print(df.head(), df.tail(), sep="\n")

#     # 3️⃣ 向量化环境：这里把 algo_name 传给 env，用于内部 logging（可随意命名）
#     env = DummyVecEnv([make_env(df, algo_name="A2C")])

#     # 4️⃣ 创建并训练 A2C 模型
#     model = A2C(
#         policy="MultiInputPolicy",  # 适配 dict obs（position, zone, zscore）
#         env=env,
#         verbose=1,
#         learning_rate=3e-4,
#         gamma=0.99,
#         n_steps=50,
#         tensorboard_log="./logs/rl1_a2c/"
#     )
#     model.learn(total_timesteps=400000)
#     model.save("models/rl1_a2c")

#         # 5️⃣ 简单评估一下最终净值（VecEnv.step 返回4项）
#     obs = env.reset()
#     done = False
#     networth = env.envs[0].networth
#     net_value_rl1 = []
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         # VecEnv.step 返回 (obs, rewards, dones, infos)
#         obs, rewards, dones, infos = env.step(action)
#         # 由于只有一个 env，所以从 array/list 里取第 0 个
#         reward = rewards[0] if hasattr(rewards, "__len__") else rewards
#         done   = dones[0]   if hasattr(dones,   "__len__") else dones
#         # 更新 networth
#         networth = env.envs[0].networth
#         net_value_rl1.append(networth)
#     print(f"▶ Final networth: {networth:.4f}")
#     # 以下动态根据 SB3 run id 保存图片
#     import matplotlib.pyplot as plt, os

#     # 取出当前 run 日志目录的最后一节（如 "A2C_14"）
#     run_dir = model.logger.dir          # e.g. "./logs/rl1_a2c/A2C_14"
#     run_id  = os.path.basename(run_dir) # -> "A2C_14"

#     # 画图并保存
#     plt.plot(net_value_rl1)
#     out_path = f"result_{run_id}.png"
#     plt.savefig(out_path)
#     print(f"▶ Equity curve saved to {out_path}")



#!/usr/bin/env python3
import os, sys
# —— 确保项目根目录在 module 路径中 —— 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from preprocess import prepare_pair_df_from_merged
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.env_rl import RL_FixAmt_PairTrade
import matplotlib.pyplot as plt


def make_env(df, algo_name):
    def _init():
        return RL_FixAmt_PairTrade(
            df=df,
            model=algo_name,    # 比如 "A2C" 或 "PPO"
            tc=0.0002,          # 0.02% 手续费
            cash=1.0,           # 初始净值 1.0
            fixed_amt=0.10,     # 每次投入净值的 10%
            verbose=1           # 打开日志
        )
    return _init


if __name__ == "__main__":
    # 1️⃣ 读入合并后的原始价格数据
    raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])

    # 2️⃣ 调用预处理：计算 spread、rolling mean/std、zscore
    df = prepare_pair_df_from_merged(raw, window=900)
    print(f"[DEBUG] df shape = {df.shape}")
    print(df.head(), df.tail(), sep="\n")

    # 3️⃣ 构造向量化环境
    # env = DummyVecEnv([make_env(df, algo_name="A2C")]) # 全量数据 
    env = DummyVecEnv([make_env(df.iloc[:400000], algo_name="A2C")]) # 只用前400000条数据train

    # 4️⃣ 创建并训练 A2C 模型
    model = A2C(
        policy="MultiInputPolicy",  # 适配 dict obs（position, zone, zscore）
        env=env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=50,
        tensorboard_log="./logs/rl1_a2c/"
    )
    model.learn(total_timesteps=400_000)
    model.save("models/rl1_a2c_train_4le5")
    model = A2C.load("models/rl1_a2c_train_4le5")
    # 5️⃣ 评估并记录净值轨迹 （包含初始点 1.0）
    # new env for test
    env = DummyVecEnv([make_env(df.iloc[1000000:1525600], algo_name="A2C")]) # 只用前400000条数据train
    obs = env.reset()

    # 将初始净值记录到列表
    net_value_rl1 = [env.envs[0].networth]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = dones[0] if isinstance(dones, (list, tuple)) else dones
        net_value_rl1.append(env.envs[0].networth)
    final_nw = env.envs[0].networth
    print(f"▶ Final networth: {final_nw:.4f}")

    # 6️⃣ 绘制并保存净值曲线，包含标题和坐标标签
    run_id = os.path.basename(model.logger.dir)  # e.g. "A2C_19"
    out_path = f"result_{run_id}.png"
    plt.figure(figsize=(8, 4))
    plt.plot(net_value_rl1)
    plt.title(f"RL1 Equity Curve ({run_id})")
    plt.xlabel("Time Steps")
    plt.ylabel("Net Worth")
    plt.savefig(out_path)
    print(f"▶ Equity curve saved to {out_path}")

import os
# 确保 result/gridsearch 目录存在，防止后续读写时报 FileNotFoundError
os.makedirs("result/gridsearch", exist_ok=True)



#!/usr/bin/env python3
import os
import pickle
import itertools
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.env_rl import RL_FixAmt_PairTrade
from preprocess import prepare_pair_df_from_merged

# 1) 读取已合并好的原始价格数据
raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])
# 只需要运行一次预处理即可复用同一个 df
base_df = prepare_pair_df_from_merged(raw, window=900)

# 2) 定义要搜索的参数范围
periods    = [600, 900, 1200]
open_thres = [1.6, 1.8, 2.0]
close_thres= [0.2, 0.4, 0.6]

best = {"period": None, "OPEN_THRE": None, "CLOS_THRE": None, "networth": -1e9}

# 3) 对每组参数，跑一次“无学习”策略（即固定动作 1——平仓），观察最终净值
for period, o, c in itertools.product(periods, open_thres, close_thres):
    # 更新 df 的 zscore 计算窗口
    df = prepare_pair_df_from_merged(raw, window=period)

    # 构造环境
    def make_env():
        return RL_FixAmt_PairTrade(
            df=df,
            model=f"GS_P{period}_O{o}_C{c}",
            tc=0.0002,
            cash=1.0,
            fixed_amt=0.1,
            verbose=0
        )
    env = DummyVecEnv([make_env])

    # 简单地“平仓”跑完全程，拿最后一次 info 里 networth
    obs, _ = env.reset()
    done = False
    last_networth = None
    while not done:
        # 这里用 action=1（平仓）做基线；如果要做 Gatev 或其他策略可自己改
        obs, _, done, _, info = env.step([1])
        last_networth = info[0].get("networth", None)

    # 记录最优
    if last_networth is not None and last_networth > best["networth"]:
        best.update(period=period, OPEN_THRE=o, CLOS_THRE=c, networth=last_networth)
        print(f"New best → period={period}, O={o}, C={c}, networth={last_networth:.4f}")

# 4) 把最优参数写入 pickle
os.makedirs("result/gridsearch", exist_ok=True)
with open("result/gridsearch/best_res.pickle", "wb") as f:
    pickle.dump({
        "period":    best["period"],
        "OPEN_THRE": best["OPEN_THRE"],
        "CLOS_THRE": best["CLOS_THRE"],
    }, f)
print("\n✅ Grid search done. Best params:", best)

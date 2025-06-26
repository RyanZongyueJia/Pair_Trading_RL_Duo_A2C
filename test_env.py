# test_env.py
import pandas as pd
from envs.env_rl import RL_FixAmt_PairTrade, RL_FreeAmt_PairTrade

if __name__ == "__main__":
    # 1) 读数据
    df = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])

    # 2) 测试 Fixed Amount 环境
    print("== Testing Fixed-Amount Env ==")
    env1 = RL_FixAmt_PairTrade(df=df, model="TEST", tc=0.0002, cash=1.0, fixed_amt=0.1, verbose=1)
    obs, info = env1.reset()
    print("Reset obs:", obs)
    for step in range(5):
        action = env1.action_space.sample()
        obs, reward, done, truncated, info = env1.step(action)
        print(f"Step {step}: action={action}, reward={reward:.4f}, done={done}")
        if done:
            break

    # 3) 测试 Free Amount 环境
    print("\n== Testing Free-Amount Env ==")
    env2 = RL_FreeAmt_PairTrade(df=df, model="TEST", tc=0.0002, cash=1.0, verbose=1)
    obs, info = env2.reset()
    print("Reset obs:", obs)
    for step in range(5):
        action = env2.action_space.sample()
        obs, reward, done, truncated, info = env2.step(action)
        print(f"Step {step}: action={action}, reward={reward:.4f}, done={done}")
        if done:
            break

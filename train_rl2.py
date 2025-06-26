# #!/usr/bin/env python3
# import os, sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# import pandas as pd
# from preprocess import prepare_pair_df_from_merged
# from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import DummyVecEnv
# from envs.env_rl import RL_FreeAmt_PairTrade

# def make_env(df, algo_name):
#     def _init():
#         return RL_FreeAmt_PairTrade(
#             df=df,
#             model=algo_name,
#             tc=0.0002,
#             cash=1.0,
#             verbose=0
#         )
#     return _init

# if __name__ == "__main__":
#     # 1) 数据准备
#     raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])
#     df  = prepare_pair_df_from_merged(raw, window=240)
#     print(f"[DEBUG] df shape = {df.shape}")

#     # 2) 环境构造
#     env = DummyVecEnv([make_env(df, algo_name="RL2_SAC")])

#     # 3) 训练 SAC
#     # model = SAC(
#     #     policy="MultiInputPolicy",
#     #     env=env,
#     #     verbose=1,
#     #     learning_rate=1e-4,
#     #     gamma=0.99,
#     #     buffer_size=100_000,
#     #     batch_size=256,
#     #     tensorboard_log="./logs/rl2_sac/"
#     # )
#     # model.learn(total_timesteps=4000)
#     # model.save("models/rl2_sac")
#     model = SAC.load("models/rl2_sac")
#     # ── 基准测试：随机策略 ──────────────────────────────
#     # obs = env.reset()
#     # done = False
#     # # 随机走一回合
#     # while not done:
#     #     action = [env.action_space.sample()]
#     #     obs, _, dones, _ = env.step(action)
#     #     done = dones[0]
#     # print("▶ Random policy networth:", env.envs[0].networth)

#     # ── 智能体评估 ──────────────────────────────
#     obs = env.reset()
#     done = False
#     # 从 env 内部读最初净值
#     networth = env.envs[0].networth
#     net_value = []
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, _, dones, _ = env.step(action)
#         done = dones[0]
#         networth = env.envs[0].networth
#         net_value.append(networth)
#     print(f"▶ RL2 learned policy networth: {networth:.4f}")
#     import matplotlib.pyplot as plt
#     plt.plot(range(len(net_value)), net_value)
#     plt.savefig("result_rl2.png")



# #!/usr/bin/env python3
# import os, sys
# # Ensure project root is in path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# import pandas as pd
# from preprocess import prepare_pair_df_from_merged
# from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import DummyVecEnv
# from envs.env_rl import RL_FreeAmt_PairTrade
# import matplotlib.pyplot as plt

# # Environment factory
# def make_env(df, algo_name):
#     def _init():
#         return RL_FreeAmt_PairTrade(
#             df=df,
#             model=algo_name,
#             tc=0.0002,
#             cash=1.0,
#             verbose=0
#         )
#     return _init

# if __name__ == "__main__":
#     # 1️⃣ Data preparation (same as RL1)
#     raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])
#     df  = prepare_pair_df_from_merged(raw, window=240)
#     print(f"[DEBUG] df shape = {df.shape}")

#     # 2️⃣ Build VecEnv
#     env = DummyVecEnv([make_env(df, algo_name="RL2_SAC")])

#     # 3️⃣ Train SAC (Free-Amt Agent)
#     model = SAC(
#         policy="MultiInputPolicy",
#         env=env,
#         verbose=1,
#         learning_rate=1e-4,
#         gamma=0.99,
#         buffer_size=500_000,
#         batch_size=256,
#         tensorboard_log="./logs/rl2_sac/"
#     )
#     model.learn(total_timesteps=100_000)
#     model.save("models/rl2_sac")

#     # Extract run ID for naming
#     run_dir = model.logger.dir            # e.g. "./logs/rl2_sac/SAC_1"
#     run_id  = os.path.basename(run_dir)   # -> "SAC_1"

#     # 4️⃣ Baseline: Random policy
#     obs = env.reset()
#     done = False
#     while not done:
#         action = [env.action_space.sample()]
#         obs, _, dones, _ = env.step(action)
#         done = dones[0] if isinstance(dones, (list, tuple)) else dones
#     print(f"▶ Random policy networth: {env.envs[0].networth:.4f}")

#     # 5️⃣ Evaluate learned policy
#     obs = env.reset()
#     done = False
#     equity = []
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, _, dones, _ = env.step(action)
#         done = dones[0] if isinstance(dones, (list, tuple)) else dones
#         equity.append(env.envs[0].networth)

#     final_nw = env.envs[0].networth
#     print(f"▶ RL2 learned policy networth: {final_nw:.4f}")

#     # 6️⃣ Plot & save equity curve
#     plt.figure(figsize=(8, 4))
#     plt.plot(equity)
#     plt.title(f"RL2 Equity Curve ({run_id})")
#     out_path = f"result_{run_id}.png"
#     plt.savefig(out_path)
#     print(f"▶ Equity curve saved to {out_path}")


# train_rl2.py


### 以下是SAC
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# import pandas as pd
# from preprocess import prepare_pair_df_from_merged
# from stable_baselines3 import SAC, A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
# from envs.env_rl import RL_FreeAmt_PairTrade
# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     # 1️⃣ 加载训练好的 RL1（A2C）模型
#     rl1_model_path = "models/rl1_A2C.zip"  # ← 改成你 RL1 实际保存的路径
#     rl1_model = A2C.load(rl1_model_path)

#     # 2️⃣ 数据准备：使用和 RL1 完全一致的 window
#     raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])
#     best_window = 900   # ← 替换成 run_gridsearch.py 中得到的最佳 period
#     df = prepare_pair_df_from_merged(raw, window=best_window)
#     print(f"[DEBUG] df shape = {df.shape}")

#     # 3️⃣ 构造向量化环境，传入 RL1 模型实例
#     def make_env():
#         return RL_FreeAmt_PairTrade(
#             df=df,
#             model=rl1_model,    # ← 真·模型实例，不是字符串
#             tc=0.0002,
#             cash=1.0,
#             verbose=0
#         )
#     env = DummyVecEnv([make_env])

#     # 4️⃣ 训练 SAC（RL2 Agent）
#     model = SAC(
#         policy="MultiInputPolicy",
#         env=env,
#         verbose=1,
#         learning_rate=1e-3,
#         gamma=0.99,
#         buffer_size=500_000,
#         batch_size=128,
#         tensorboard_log="./logs/rl2_sac/"
#     )
#     total_timesteps = 400_000
#     model.learn(total_timesteps=total_timesteps)
#     model.save("models/rl2_sac")  # 会保存成 models/rl2_sac.zip

#     # 从日志目录提取 run_id，用于命名图表
#     run_id = os.path.basename(model.logger.dir)

#     # # 5️⃣ Baseline：随机策略净值
#     # obs = env.reset()
#     # done = False
#     # while not done:
#     #     action = [env.action_space.sample()]
#     #     obs, _, dones, _ = env.step(action)
#     #     done = dones[0] if isinstance(dones, (list, tuple)) else dones
#     # print(f"▶ Random policy networth: {env.envs[0].networth:.4f}")

#     # 6️⃣ 评估 SAC 策略并记录净值曲线
#     obs = env.reset()
#     done = False
#     equity = []
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, _, dones, _ = env.step(action)
#         done = dones[0] if isinstance(dones, (list, tuple)) else dones
#         equity.append(env.envs[0].networth)
#     print(f"▶ RL2 learned policy networth: {env.envs[0].networth:.4f}")

#     # 7️⃣ 绘制并保存净值曲线
#     plt.figure(figsize=(8, 4))
#     plt.plot(equity)
#     plt.title(f"RL2 Equity Curve ({run_id})")
#     out_path = f"result_{run_id}.png"
#     plt.savefig(out_path)
#     print(f"▶ Equity curve saved to {out_path}")


#!/u#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from preprocess import prepare_pair_df_from_merged
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.env_rl import RL_FreeAmt_PairTrade
import matplotlib.pyplot as plt


def make_env(df, rl1_model):
    def _init():
        return RL_FreeAmt_PairTrade(
            df=df,
            model=rl1_model,
            tc=0.0002,
            cash=1.0,
            verbose=1
        )
    return _init


if __name__ == "__main__":
    # 1️⃣ 加载已训练的 RL1 模型（A2C）
    from stable_baselines3 import A2C as _A2C
    rl1_model = _A2C.load("models/rl1_a2c_train_4le5.zip")

    # 2️⃣ 数据准备，使用与 RL1 相同的 window
    raw = pd.read_csv("data/processed_pair.csv", parse_dates=["datetime"])
    df = prepare_pair_df_from_merged(raw, window=900)
    print(f"[DEBUG] total df shape = {df.shape}")

    # 3️⃣ 划分训练/测试集
    train_end = 400_000
    test_end  = train_end + 525600  # 一年=525600分钟 测试数据
    df_train = df.iloc[:train_end]
    df_test  = df.iloc[train_end:test_end]
    print(f"[DEBUG] train df: {df_train.shape}, test df: {df_test.shape}")

    # 4️⃣ 构造训练环境
    train_env = DummyVecEnv([make_env(df_train, rl1_model)])

    # 5️⃣ 创建并训练 RL2 A2C 模型
    model = A2C(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=50,
        tensorboard_log="./logs/rl2_a2c_train_4le5/"
    )
    total_timesteps = 400_000
    model.learn(total_timesteps=total_timesteps)
    model.save("models/rl2_a2c_train_4le5")

    # 6️⃣ 构造测试环境并评估
    test_env = DummyVecEnv([make_env(df_test, rl1_model)])
    obs = test_env.reset()
    equity = [test_env.envs[0].networth]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = test_env.step(action)
        done = dones[0] if isinstance(dones, (list, tuple)) else dones
        equity.append(test_env.envs[0].networth)
    final_nw = test_env.envs[0].networth
    print(f"▶ Test final networth: {final_nw:.4f}")

    # 7️⃣ 保存测试净值曲线
    run_id = os.path.basename(model.logger.dir)
    plt.figure(figsize=(8, 4))
    plt.plot(equity)
    plt.title(f"RL2 Equity Curve A2C ({run_id})")
    plt.xlabel("Time Steps")
    plt.ylabel("Net Worth")
    out_file = f"result_RL2_{run_id}.png"
    plt.savefig(out_file)
    print(f"▶ Equity curve saved to {out_file}")

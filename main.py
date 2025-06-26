#!/usr/bin/env python3
import argparse
import pandas as pd
from envs.env_gridsearch import PairTrading  # 你的网格搜索
# from train_rl1 import train_rl1             # RL1：定时决策
# from train_rl2 import train_rl2             # RL2：定时+规模决策
from envs.mock_trading import TradingSystemFixAmt, TradingSystemFreeAmt


if __name__ == '__main__':
    df = pd.read_csv("data/data.csv", header=None)

    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['grid','rl1','rl2','backtest'], required=True)
    p.add_argument('--df1', help='BTC_EUR CSV 路径')
    p.add_argument('--df2', help='BTC_GBP CSV 路径')
    p.add_argument('--algo', choices=['A2C','PPO','DQN','SAC'], default='A2C')
    p.add_argument('--window', type=int, default=900)
    p.add_argument('--open_thres', type=float, default=1.8)
    p.add_argument('--close_thres', type=float, default=0.4)
    args = p.parse_args()

    # 载入数据
    df1 = pd.read_csv(args.df1, parse_dates=['timestamp'])
    df2 = pd.read_csv(args.df2, parse_dates=['timestamp'])

    if args.mode == 'grid':
        run_gridsearch(df1, df2, windows=[500,900,1200], opens=[1.6,1.8,2.0], closes=[0.3,0.4,0.5])
    elif args.mode == 'rl1':
        train_rl1(df1, df2, algo=args.algo, window=args.window, open_thres=args.open_thres, close_thres=args.close_thres)
    elif args.mode == 'rl2':
        train_rl2(df1, df2, algo=args.algo, window=args.window, open_thres=args.open_thres, close_thres=args.close_thres)
    elif args.mode == 'backtest':
        backtest('models/rl2_A2C', df1, df2, window=args.window, open_thres=args.open_thres, close_thres=args.close_thres)

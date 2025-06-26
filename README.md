# Reinforcement Learning Pair Trading
# 强化学习配对交易复现  

本项目复现并扩展了 Yang & Malik (2024) 提出的“动态尺度强化学习配对交易”方法[^1]，包含两阶段代理：  
- **RL1 (Fixed Amount Agent)**：学习“何时开/平仓”，每次固定投入比例  
- **RL2 (Free Amount Agent)**：基于 RL1 信号，动态学习“投入规模”  

This project reproduces and extends the “Dynamic Scaling Reinforcement Learning Pair Trading” approach proposed by Yang & Malik (2024)[^1], featuring a two-stage agent design:  
- **RL1 (Fixed Amount Agent)**: learns “when to enter/exit” with a fixed trade size  
- **RL2 (Free Amount Agent)**: learns “how much to trade” dynamically on top of RL1 signals  

---
Pair_Trading_RL_Duo_A2C/
├── data/                         # 原始和预处理后数据(This is a way smaller dataset than what we need to train and run this thing)
│   ├── BTCEUR.csv
│   ├── BTCGBP.csv
│   └── processed_pair.csv
├── envs/                         # 环境定义（复用 Yang 同学开源代码）
│   ├── env_rl.py
│   ├── mock_trading.py
│   └── LICENSE                   # 原环境作者 MIT 许可证（© 2023 Hongshen Yang）
├── utils/                        # 本项目工具函数
│   └── logger.py
├── logs/                         # 训练/评估日志（TensorBoard）
├── models/                       # 已训练的 RL1/RL2 模型文件
├── preprocess.py                 # 原始数据预处理脚本
├── process_data.py               # 生成 processed_pair.csv 的脚本
├── run_gridsearch.py             # 网格搜索 OPEN_THRE/CLOS_THRE/period
├── train_rl1.py                  # 训练 RL1（Fixed Amount Agent）
├── train_rl2_simple.py           # 全量数据训练 RL2（Free Amount Agent）
├── train_rl2_partial.py          # 划分训练/测试集训练 RL2
├── environment.yml               # Conda 环境配置
├── .gitignore                    # 忽略文件列表
├── LICENSE                       # 本项目 MIT 许可证（© 2025 Ryan Zongyue Jia）
└── README.md                     # 项目说明文档





环境 / Environment
克隆仓库 / Clone:

git clone (https://github.com/RyanZongyueJia/Pair_Trading_RL_Duo_A2C.git)

创建并激活 Conda 环境 / Create & activate Conda env:

conda env create -f environment.yml
conda activate pair-trading
验证依赖版本 / Verify key packages:


python -c "import pandas, gymnasium, stable_baselines3; print(pandas.__version__)"
数据预处理 / Data Preprocessing
将原始 CSV（data/BTCEUR.csv、data/BTCGBP.csv）转换为训练输入：

python process_data.py
生成 data/processed_pair.csv，包含：

datetime, close0, close1
spread, rolling_mean, rolling_std, zscore

超参数搜索 / Hyperparameter Search
在 processed_pair.csv 上做网格搜索，得到最优的：

period (window)
OPEN_THRE, CLOS_THRE

python run_gridsearch.py
结果保存在 result/gridsearch/best_res.pickle，脚本会自动读取。

RL1 训练 / RL1 Training
使用最优 period 训练 Fixed-Amount Agent：

使用 A2C 算法

每次固定投入净值的 10%

训练结束后模型保存在 models/rl1_a2c.zip

equity curve（含初始净值 1.0）保存在 result_A2C_*.png

RL2 训练 / RL2 Training

用 A2C 训练 RL2，模型保存在 models/rl2_a2c_*.zip

equity curve 保存在 result_RL2_*.png

结果展示 / Results
所有 equity curve 图像位于项目根目录，模型文件保存在 models/，日志在 logs/ 目录。



## 致谢 / Acknowledgements

本项目复用了 Yang 在 GitHub 上开源的环境代码（`envs/` 目录），
原仓库：https://github.com/Hongshen-Yang/pair-trading-envs  
请参阅 `LICENSE-ENVS`（MIT License, © 2023 Yang）了解原作者许可。


### Rule-based Pair Trading Environment
Rule-based pair trading environments with [backtrader](https://www.backtrader.com/) framework (`env_gridsearch.ipynb`)

### Gymnasium-based Pair Trading Environment
Reinforcement Learning based environment with [gymnasium](https://gymnasium.farama.org/index.html)  (`env_rl.ipynb`)
* Fixed Amount: The bet for each trading is fixed at a certain number.
* Free Amount: The bet is dynamically measured by the each trading opportunity.

[^1]: Yang, H., & Malik, A. (2024). Reinforcement learning pair trading: A dynamic scaling approach. Journal of Risk and Financial Management, 17(12), 555. https://doi.org/10.3390/jrfm17120555



import pickle
import gymnasium as gym
import numpy as np
import pandas as pd

from envs.mock_trading import TradingSystemFixAmt, TradingSystemFreeAmt
from utils.logger import logger

def read_best_params():
    with open('result/gridsearch/best_res.pickle', 'rb') as pk:
        _, best_params = pickle.load(pk)
    return best_params

class RL_FixAmt_PairTrade(gym.Env):
    def __init__(self, df, model='', tc=0.0002, cash=1.0, fixed_amt=0.1, verbose=0):
        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.Discrete(3), # {0, 1, 2}
                    #   Position 0: shorting leg_0 -> longing leg_1
                    #   Position 1:         empty holding
                    #   Position 2: longing leg_0 <- shorting leg_1
            'zone':  gym.spaces.Discrete(5), # {0, 1, 2, 3, 4}
                    # The zscore comes from price0-price1, zone0 stands for price0 way higher than price1
                    #   Zone 0 (Should be position 0)
                    # ---------- + OPEN_THRES ----------
                    #   Zone 1 (Should be position 0, 1)
                    # ---------- + CLOS_THRES ----------
                    #   Zone 2 (Should be position 1)
                    # ----------   ZSCORE = 0 ----------
                    #   Zone 2 (Should be position 1)
                    # ---------- - CLOS_THRES ----------
                    #   Zone 3 (Should be position 1, 2)
                    # ---------- - OPEN_THRES ----------
                    #   Zone 4 (Should be position 2)
            'zscore': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64)
        })
        self.action_space = gym.spaces.Discrete(3) # {0: "short leg0 long leg1", 1: "close positions", 2: "long leg0 short leg1"}

        self.verbose = verbose
        self.cash, self.networth = cash, cash
        self.fixed_amt = fixed_amt
        self.df = df
        self.model = model
        self.best_params = read_best_params()
        self.holdings = [0, 0] #[400, -300] That means we have 400 unit of leg0 and -300 unit of leg1



    def _get_obs(self):
        # clamp 索引，保证不越界
        idx = min(self.trade_step, len(self.df) - 1)
        zscore = self.df.iloc[idx]['zscore']


        if zscore > self.best_params['OPEN_THRE']:
            zone = 0
        elif zscore > self.best_params['CLOS_THRE']:
            zone = 1
        elif zscore < -self.best_params['OPEN_THRE']:
            zone = 4
        elif zscore < -self.best_params['CLOS_THRE']:
            zone = 3
        else:
            zone = 2

        obs = {
            'position': self.position,
            'zone': zone,
            'zscore': np.array([zscore])
        }

        return obs
    
    def _get_reward(self, prev_networth):
        act_rwd = 1
        
        if self.signal['zone']==0 and self.signal['position']==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==0 and self.signal['position']==1:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==0 and self.signal['position']==2:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['position']==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['position']==1:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==1 and self.signal['position']==2:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==0:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==1:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==2:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==0:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==1:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==2:
            reward = act_rwd if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==0:
            reward = act_rwd if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==1:
            reward = act_rwd if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==2:
            reward = act_rwd if self.action==2 else 0

        reward += (self.networth - prev_networth)*100
        return reward

    def _take_action(self):
        sys=TradingSystemFixAmt(self.df, self.holdings, self.trade_step, cash=self.cash, amt=self.fixed_amt)

        if self.position==0 and self.action==0:
            # Do nothing
            pass
        elif self.position==0 and self.action==1:
            # Close position
            self.cash, self.holdings = sys.close_position()
            self.networth = sys.get_networth()
        elif self.position==0 and self.action==2:
            # Long leg0 short leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==1 and self.action==0:
            # Short leg0 long leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==1 and self.action==1:
            # Do nothing
            pass
        elif self.position==1 and self.action==2:
            # Long leg0 short leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==2 and self.action==0:
            # Short leg0 long leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==2 and self.action==1:
            # Close position
            self.cash, self.holdings = sys.close_position()
            self.networth = sys.get_networth()
        elif self.position==2 and self.action==2:
            # Do nothing
            pass

        self.position = self.action

    def reset(self, seed=None):
        self.position = 1
        self.trade_step = self.best_params['period']
        self.observation = self._get_obs()
        return self.observation, {}

    def step(self, action):
        self.action = action
        self.signal = self.observation
        prev_networth = self.networth
        self._take_action()
        self.trade_step += 1
        self.observation = self._get_obs()
        terminated = self.trade_step >= len(self.df)
        truncated = False
        self.reward = self._get_reward(prev_networth)

        if self.verbose==1:
            curr_idx = min(self.trade_step, len(self.df)-1)
            curr_df = self.df.iloc[curr_idx]
            
            logger(self.model, curr_df['datetime'], self.networth, self.action, curr_df['zscore'], self.position, curr_df['close0'], curr_df['close1'])

        return self.observation, self.reward, terminated, truncated, {}

    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{round(self.reward, 3)}, networth: {round(self.networth, 4)}")

    def close(self):
        print("Finished")
        print(f"networth: {self.networth}")

class RL_FreeAmt_PairTrade(gym.Env):
    def __init__(self, df, model='', tc=0.0002, cash=1.0, verbose=0): # act_pun is action punishment
        self.observation_space = gym.spaces.Dict({
            'holdings': gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32), # {0, 1, 2}
                    #   Position=0: position closed
                    #   Position>0: longing
                    #   Position<0: shorting
            'zone':  gym.spaces.Discrete(5), # {0, 1, 2, 3, 4}
                    # The zscore comes from price0-price1, zone0 stands for price0 way higher than price1
                    #   Zone 0
                    # ---------- + OPEN_THRES ----------
                    #   Zone 1
                    # ---------- + CLOS_THRES ----------
                    #   Zone 2
                    # ----------   ZSCORE = 0 ----------
                    #   Zone 2
                    # ---------- - CLOS_THRES ----------
                    #   Zone 3
                    # ---------- - OPEN_THRES ----------
                    #   Zone 4
            'zscore': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64)
        })

        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float64)
        # {[-1, 0]: "short leg0 long leg1", 0: "close positions", [0,1]: "long leg0 short leg1"}

        self.verbose = verbose
        self.cash, self.networth = cash, cash
        self.df = df
        self.model = model
        self.best_params = read_best_params()
        self.holdings = np.array([0], dtype=np.float32) #[1, -1] That means we have 1 unit of leg0 and -1 unit of leg1
        self.units = np.array([0, 0], dtype=np.float32) # Holding but in units
        self.tc = tc

    def _get_obs(self):
        zscore = self.df.iloc[self.trade_step]['zscore']

        if zscore > self.best_params['OPEN_THRE']:
            zone = 0
        elif zscore > self.best_params['CLOS_THRE']:
            zone = 1
        elif zscore < -self.best_params['OPEN_THRE']:
            zone = 4
        elif zscore < -self.best_params['CLOS_THRE']:
            zone = 3
        else:
            zone = 2
        
        df_current = self.df.iloc[self.trade_step]
        price0 = df_current['close0']
        value0= self.units[0]*price0
        perc = value0/self.networth
        self.holdings = np.array([perc], dtype=np.float32)

        obs = {
            'holdings': self.holdings,
            'zone': zone,
            'zscore': np.array([zscore], dtype=np.float32)
        }

        return obs
    
    def _get_reward(self, prev_networth):

        act_rwd = 1
        act_rwd_lvl1 = 1 # Close a position in the right time
        act_rwd_lvl2 = 0.7 # Open a position in the right time
        act_rwd_lvl3 = 0.5 # Do nothing in the right time

        if   self.signal['zone']==0 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==0 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==0 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action<=0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action>=0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action>0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action>0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action>0 else 0

        # Transaction Fee Punishment
        act_pnm = abs(self.action-self.signal['holdings'][0])/50
        reward -= act_pnm

        reward += (self.networth-prev_networth)*100

        # reward += 0.1 if self.action==0 else 0
        
        return reward

    def _take_action(self):
        sys=TradingSystemFreeAmt(
            self.df, self.units, self.trade_step, self.cash, self.tc)

        if   self.holdings[0]<0 and self.action<0:
            self.cash, self.units = sys.adjust_position(self.action)
        elif self.holdings[0]<0 and self.action==0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = self.cash
        elif self.holdings[0]<0 and self.action>0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = self.cash
            # self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]==0 and self.action<0:
            # Open position
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]==0 and self.action==0:
            # Do nothing
            pass
        elif self.holdings[0]==0 and self.action>0:
            # Open position
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]>0 and self.action<0:
            # Close and open position
            self.cash, self.units = sys.close_position()
            self.networth = self.cash
            # self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]>0 and self.action==0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = self.cash
        elif self.holdings[0]>0 and self.action>0:
            self.cash, self.units = sys.adjust_position(self.action)
        
    def reset(self, seed=None):
        self.holdings = np.array([0], dtype=np.float32)
        self.units = np.array([0, 0], dtype=np.float32)
        self.trade_step = self.best_params['period']
        self.observation = self._get_obs()
        return self.observation, {}

    def step(self, action):
        self.action = action[0]
        # Separate Signal as previous observation
        self.signal = self.observation
        prev_networth = self.networth
        self._take_action()
        self.trade_step += 1
        
        done = self.trade_step >= len(self.df) - 1

        self.observation = self._get_obs()
        terminated = done
        truncated = False
        self.reward = self._get_reward(prev_networth)

        if self.verbose==1:
            curr_df = self.df.iloc[self.trade_step]
            logger(self.model, curr_df['datetime'], self.networth, self.action, curr_df['zscore'], self.holdings, curr_df['close0'], curr_df['close1'])

        return self.observation, self.reward, terminated, truncated, {}

    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{round(self.reward, 3)}, networth: {round(self.networth, 4)}")

    def close(self):
        print("Finished")
        print(f"networth: {self.networth}")
        

# class RL_FreeAmt_PairTrade(gym.Env):
#     """
#     A Gym environment for RL2: free-amount pair trading.
#     Leverages a pre-trained RL1 model for trading signals, then determines order size.
#     """
#     def __init__(self, df, model='', tc=0.0002, cash=1.0, verbose=0):
#         # 1️⃣ Spaces
#         self.observation_space = gym.spaces.Dict({
#             'holdings': gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32),
#             'zone':     gym.spaces.Discrete(5),
#             'zscore':   gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
#         })
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32)

#         # 2️⃣ Market data
#         self.df = df
#         self.tc = tc
#         self.initial_cash = cash
#         self.cash = cash
#         self.networth = cash
#         self.verbose = verbose
#         self.best_params = read_best_params()

#         # 3️⃣ Load pre-trained RL1 model (A2C)
#         if isinstance(model, str) and model.endswith('.zip'):
#             from stable_baselines3 import A2C
#             self.rl1_model = A2C.load(model)
#         else:
#             self.rl1_model = model

#         # 4️⃣ Internal state placeholders
#         self.units = np.array([0.0, 0.0], dtype=np.float32)
#         self.holdings = np.array([0.0], dtype=np.float32)
#         self.trade_step = 0

#         # 5️⃣ Action thresholding to avoid micro-trades
#         self.min_trade_amt = 0.01

#         # 6️⃣ Initialize first observation
#         self.observation = self._get_obs()
#         self.signal = self.observation
#         self.action = 0.0

#     def _get_obs(self):
#         idx = min(self.trade_step, len(self.df)-1)
#         row = self.df.iloc[idx]
#         zscore = float(row['zscore'])
#         # Determine zone by best_params
#         if zscore > self.best_params['OPEN_THRE']:
#             zone = 0
#         elif zscore > self.best_params['CLOS_THRE']:
#             zone = 1
#         elif zscore < -self.best_params['OPEN_THRE']:
#             zone = 4
#         elif zscore < -self.best_params['CLOS_THRE']:
#             zone = 3
#         else:
#             zone = 2
#         # Compute current holding pct of leg0 for state
#         price0 = float(row['close0'])
#         self.networth = self.cash + self.units[0] * price0
#         holding_pct = (self.units[0]*price0) / self.networth if self.networth > 0 else 0.0
#         self.holdings = np.array([holding_pct], dtype=np.float32)

#         return {
#             'holdings': self.holdings,
#             'zone': zone,
#             'zscore': np.array([zscore], dtype=np.float32)
#         }

#     # def _get_reward(self, prev_networth):
#     #     # Reward = networth change *100 minus micro-trade penalty
#     #     delta = self.networth - prev_networth
#     #     micro_pen = abs(self.action - self.signal['holdings'][0]) / 50
#     #     return float(delta * 100 - micro_pen)
#     def _get_reward(self, prev_networth):
#         # ① networth 差分放大
#         delta = self.networth - prev_networth
#         scaled_delta = delta * 1_000     # 放大系数从 100 → 1000

#         # ② 微交易惩罚减弱
#         micro_pen = abs(self.action - self.signal['holdings'][0]) / 100  # 从 /50 → /100

#         return float(scaled_delta - micro_pen)

#     def _take_action(self, eff_action):
#         ts = TradingSystemFreeAmt(
#             self.df, self.units, self.trade_step, self.cash, self.tc
#         )
#         # No action
#         if eff_action == 0.0:
#             return
#         # Determine directions
#         curr_dir = np.sign(self.holdings[0])
#         new_dir  = np.sign(eff_action)
#         # Close if switching side
#         if curr_dir != 0 and new_dir != curr_dir:
#             self.cash, self.units = ts.close_position()
#         # Open or adjust position if any
#         qty = abs(eff_action)
#         if new_dir == 0:
#             return
#         if curr_dir == 0:
#             self.cash, self.units = ts.open_position(new_dir * qty)
#         else:
#             self.cash, self.units = ts.adjust_position(new_dir * qty)
#         # Update networth after trade
#         price0 = float(self.df.iloc[self.trade_step]['close0'])
#         self.networth = self.cash + self.units[0]*price0

#     def reset(self, seed=None):
#         self.cash = self.networth = self.initial_cash
#         self.units = np.array([0.0, 0.0], dtype=np.float32)
#         self.holdings = np.array([0.0], dtype=np.float32)
#         self.trade_step = 0
#         obs = self._get_obs()
#         self.observation = obs
#         return obs, {}

#     def step(self, action):
#         # Build RL2 observation first
#         mag = float(action[0])
#         # Threshold micro actions
#         if abs(mag) < self.min_trade_amt:
#             mag = 0.0
#         # Build RL1 observation (position, zone, zscore)
#         hold_pct = self.holdings[0]
#         if hold_pct < 0:
#             pos = 0
#         elif hold_pct > 0:
#             pos = 2
#         else:
#             pos = 1
#         rl1_obs = {
#             'position': np.array([pos], dtype=np.int64),
#             'zone': pos and self.observation['zone'] or self.observation['zone'],
#             'zscore': self.observation['zscore']
#         }
#         # Get RL1 signal
#         sig, _ = self.rl1_model.predict(rl1_obs, deterministic=True)
#         sig = float(sig)
#         # Effective order size
#         eff_action = mag * sig
#         prev_nw = self.networth
#         self.signal = self.observation
#         self.action = eff_action
#         # Execute trade
#         self._take_action(eff_action)
#         # Advance step
#         self.trade_step += 1
#         done = self.trade_step >= len(self.df)-1
#         obs = self._get_obs()
#         self.observation = obs
#         reward = self._get_reward(prev_nw)
#         if self.verbose:
#             row = self.df.iloc[min(self.trade_step, len(self.df)-1)]
#             logger(self.rl1_model, row['datetime'], self.networth,
#                    eff_action, row['zscore'], self.holdings,
#                    row['close0'], row['close1'])
#         return obs, reward, done, False, {}

#     def render(self):
#         print(f"Step {self.trade_step} | action={self.action:.4f} | networth={self.networth:.4f}")

#     def close(self):
#         print(f"Finished, final networth = {self.networth:.4f}")

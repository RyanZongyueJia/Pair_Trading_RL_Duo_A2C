import pandas as pd

def prepare_pair_df_from_merged(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    # —— 新增，去掉千分位逗号并转 float —— 
    df['close0'] = df['close0'].astype(str).str.replace(',', '').astype(float)
    df['close1'] = df['close1'].astype(str).str.replace(',', '').astype(float)

    # 接下来计算 spread / rolling / zscore
    df['spread'] = df['close0'] - df['close1']
    df['ma']     = df['spread'].rolling(window).mean()
    df['std']    = df['spread'].rolling(window).std()
    df = df.dropna().reset_index(drop=True)
    df['zscore'] = (df['spread'] - df['ma']) / df['std']

    return df[['datetime','close0','close1','spread','zscore']]

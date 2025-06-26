import os
import pandas as pd

def process_btc_pair(
    output_path="data/processed_pair.csv",
    window=240
):
    # 定位项目根目录，保证无论你从哪跑都能找到 data 文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eur_path = os.path.join(base_dir, "data", "BTCEUR.csv")
    gbp_path = os.path.join(base_dir, "data", "BTCGBP.csv")

    # 1) 读入，用第二行当表头
    df_eur = pd.read_csv(eur_path, sep="|")
    df_gbp = pd.read_csv(gbp_path, sep="|")

    # 2) 剥除列名空白，打印确认
    df_eur.columns = df_eur.columns.str.strip()
    df_gbp.columns = df_gbp.columns.str.strip()
    print("EUR columns:", df_eur.columns.tolist())
    print("GBP columns:", df_gbp.columns.tolist())

    # 3) 时间戳转 datetime
    df_eur["datetime"] = pd.to_datetime(df_eur["Open timestamp"], unit="s")
    df_gbp["datetime"] = pd.to_datetime(df_gbp["Open timestamp"], unit="s")

    # 4) 提取收盘价并重命名
    df_eur = df_eur[["datetime", "Close"]].rename(columns={"Close": "close0"})
    df_gbp = df_gbp[["datetime", "Close"]].rename(columns={"Close": "close1"})

    # 5) 按时间做 inner join
    df = pd.merge(df_eur, df_gbp, on="datetime", how="inner")

    # 6) 计算 spread 和 rolling z-score
    df["spread"] = df["close0"] - df["close1"]
    roll = df["spread"].rolling(window=window)
    df["zscore"] = (df["spread"] - roll.mean()) / roll.std()

    # 7) 丢掉前 window 行的 NaN 并保存
    df = df.dropna().reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.join(base_dir, output_path)), exist_ok=True)
    df.to_csv(os.path.join(base_dir, output_path), index=False)
    print(f"✔ saved {output_path}, rows = {len(df)}")

if __name__ == "__main__":
    process_btc_pair()

# %% [markdown]
# ## 通过akshare获取当日股票数据 并 存入pickle文件

# %%
import akshare as ak
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from pandas import DataFrame, Series
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import pickle
import os

import yfinance as yf

import common
import calculate

# %%
def get_today_stock_data():
    """
    获取每日股票date open high low close 数据
    可在 限制股票数量 进行测试
    """
    file_name = "./txt_lib/stock_code.txt"
    with open(file_name, "r") as file:
        stock_list = [line.strip() for line in file if line.strip()]

    # # 限制股票数量
    # stock_list = stock_list[:150]

    def get_stock_data(stock_code: str):
        """Fetch daily stock data for a stock code"""

        df = common.Download.download_with_retry(stock_code)

        # Convert date columns to strings
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        return stock_code, df.to_dict()

    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(get_stock_data, stock_list))

    # Convert results to a dictionary with stock codes as keys and DataFrame dictionaries as values
    data = {stock_code: df for stock_code, df in results}

    return data


def create_pickle(data: dict):
    """先移除前一个工作日的pickle文件,再生成今日的pickle文件 (避免permission error报错)"""
    pickle_file = "./txt_lib/daily_df.pkl"
    os.remove(pickle_file)
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)

    print("Stock data has been saved to './txt_lib/daily_df.pkl'.")

# %%
data = get_today_stock_data()
create_pickle(data)

# %% [markdown]
# ## TEST

# %%
# from importlib import reload

# reload(calculate)
# reload(common)




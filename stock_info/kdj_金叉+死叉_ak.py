# %% [markdown]
# ## 一周内周KDJ金叉，期间日KDJ金叉且无死叉

# %% [markdown]
# ## 一周内周KDJ金叉，当日日KDJ死叉

# %%
import akshare as ak
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import os

import common
import calculate

current_date = datetime.now() + timedelta(days=1)
one_week_ago = current_date - timedelta(days=7)  # 计算一周之前的日期

# 计算KDJ指标
def calculate_kdj(df: DataFrame, n=9):
    """计算 KDJ 指标"""
    low_min = df["low"].rolling(window=n).min()
    high_max = df["high"].rolling(window=n).max()
    rsv = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["K"] = rsv.ewm(com=2).mean()  # K线
    df["D"] = df["K"].ewm(com=2).mean()  # D线
    df["J"] = 3 * df["K"] - 2 * df["D"]  # J线
    return df

# 按周重新采样，重新计算周K线数据的KDJ
def calculate_weekly_kdj(df: DataFrame):
    """计算周K线的KDJ指标"""
    df["date"] = pd.to_datetime(df["date"])  # 确保 date 列是 datetime 格式
    df = df.set_index("date")  # 把 date 设为索引

    weekly_data = (
        df.resample("W-FRI")  # 取当周五的数据
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    # 获取当前日期
    today = pd.Timestamp.today().normalize()  # 只取日期部分，去掉时间
    last_weekly_index = weekly_data.index[-1]  # 取最后一个周五索引

    # 如果今天不是周五，修改最后一行索引为今天
    if today != last_weekly_index:
        weekly_data = weekly_data.rename(index={last_weekly_index: today})

    return calculate_kdj(weekly_data)


def find_kdj_golden_cross(df: DataFrame):
    """查找 KDJ 金叉和死叉，并判断当日股票涨跌"""
    df["Golden_Cross"] = (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))
    df["Dead_Cross"] = (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))

    # 判断股票涨跌：今日收盘价高于昨日收盘价则为上涨（Up），否则为下跌（Down）
    df["Price_Change"] = df["close"].diff()  # 计算收盘价变化
    df["Trend"] = df["Price_Change"].apply(lambda x: "Up" if x > 0 else "Down")

    return df


# 查找最近金叉时段 且 必须是1最近一周
def get_golden_to_dead_cross_periods(df: DataFrame):
    periods = []
    start_date = None  # 用于记录金叉的开始日期

    for i in range(len(df)):
        if df["Golden_Cross"].iloc[i]:  # 如果是金叉
            if start_date is None:  # 开始记录金叉日期
                start_date = df.index[i]
        elif df["Dead_Cross"].iloc[i]:  # 如果是死叉
            if start_date is not None:  # 如果已记录金叉
                end_date = df.index[i]
                # 不再保存金叉-死叉日期段
                # periods.append((start_date, end_date))  # 保存当前金叉-死叉日期段
                start_date = None  # 重置金叉开始日期

    # 如果循环结束时还有未匹配的金叉（没有死叉）
    if start_date is not None:
        # 检查是否小于7天
        is_within_seven_days = ((df.index[-1] - start_date)).days <= 7

        if is_within_seven_days:
            periods.append((start_date, df.index[-1]))  # 配对到最后一行的日期

    return periods  # 返回金叉日期段


def filter_golden_cross_without_dead_cross(data: DataFrame, periods: list):

    # 提取金叉日期
    golden_cross_dates = data[data["Golden_Cross"]]

    # 检查周 KDJ 的金叉段内，无死叉条件
    filtered_dates = []

    # 查找最近日线kdj金叉日期
    last_golden_cross_date = golden_cross_dates.iloc[-1]["date"]

    # 判断和今日时间差
    is_recent = (current_date - last_golden_cross_date).days < 2

    if periods and is_recent:    
        filtered_dates.append(last_golden_cross_date)

    # 判断今日股票涨跌的
    if data["Trend"].iloc[-1] == "Down":
        return []

    return filtered_dates


# 查找符合的股票
def get_recent_golden_cross_dates(stock_code: str):
    data = download_with_retry(stock_code)

    if data.empty:  # 检查是否下载失败
        print(f"Failed to download data for {stock_code}. Skipping...")
        return []

    # 换成ak了，不需要去掉第二层的ticker
    # data.columns = data.columns.droplevel(1)  # 去掉第二层的 ticker，变为单层索引

    data = calculate_kdj(data)  # 计算日 KDJ
    weekly_data = calculate_weekly_kdj(data)  # 计算周 KDJ
    weekly_data = find_kdj_golden_cross(weekly_data)  # 查找周 K线金叉
    data = find_kdj_golden_cross(data)  # 查找日线 KDJ 金叉和死叉
    periods = get_golden_to_dead_cross_periods(
        weekly_data
    )  # 提取周 KDJ 金叉到死叉的日期段

    # 获取符合条件的金叉日期
    filtered_golden_cross_dates = filter_golden_cross_without_dead_cross(data, periods)

    # 将 filtered_golden_cross_dates 转换为 pandas 的 DatetimeIndex 以便筛选
    filtered_golden_cross_dates = pd.to_datetime(filtered_golden_cross_dates)

    # 筛选出最近一周内的日期
    recent_dates = [
        date.strftime("%Y-%m-%d")
        for date in filtered_golden_cross_dates
        if one_week_ago <= date <= current_date
    ]

    return recent_dates


# 查找周kdj金叉 + 当日kdj死叉
def filter_golden_cross_with_dead_cross(data: DataFrame, periods: list):
    filtered_dates = []
    if periods:
        # 检查今日是否出现死叉
        today_dead_cross = data["Dead_Cross"].iloc[-1]
        if today_dead_cross:
            filtered_dates.append(current_date)

    return filtered_dates


# 查找符合的股票
def get_recent_death_cross_dates(stock_code: str):
    data = download_with_retry(stock_code)

    if data.empty:  # 检查是否下载失败
        print(f"Failed to download data for {stock_code}. Skipping...")
        return []

    # 换成ak了，不需要去掉第二层的ticker
    # data.columns = data.columns.droplevel(1)  # 去掉第二层的 ticker，变为单层索引

    data = calculate_kdj(data)  # 计算日 KDJ
    weekly_data = calculate_weekly_kdj(data)  # 计算周 KDJ
    weekly_data = find_kdj_golden_cross(weekly_data)  # 查找周 K线金叉
    data = find_kdj_golden_cross(data)  # 查找日线 KDJ 金叉和死叉
    periods = get_golden_to_dead_cross_periods(
        weekly_data
    )  # 提取周 KDJ 金叉到死叉的日期段

    # 获取符合条件的金叉日期
    filtered_golden_cross_dates = filter_golden_cross_with_dead_cross(data, periods)

    # 将 filtered_golden_cross_dates 转换为 pandas 的 DatetimeIndex 以便筛选
    filtered_golden_cross_dates = pd.to_datetime(filtered_golden_cross_dates)

    # 筛选出最近一周内的日期
    recent_dates = [
        date.strftime("%Y-%m-%d")
        for date in filtered_golden_cross_dates
        if one_week_ago <= date <= current_date
    ]

    return recent_dates

def process_stock(stock_code: str):
    """处理单个股票，获取金叉和死叉日期"""
    time.sleep(random.uniform(0.1, 0.2))  # 随机延时
    try:
        golden_cross_dates = get_recent_golden_cross_dates(stock_code)
        death_cross_dates = get_recent_death_cross_dates(stock_code)
        return stock_code, golden_cross_dates, death_cross_dates
    except Exception as e:
        print(f"Error processing {stock_code}: {e}")
        return stock_code, [], []


def process_stocks(stock_list: list):
    """多线程处理股票列表，获取所有股票的金叉和死叉日期"""
    all_cross_dates = {}  # 存储所有股票的交叉日期
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {
            executor.submit(process_stock, stock): stock for stock in stock_list
        }

        for future in as_completed(future_to_stock):
            stock_code, golden_dates, death_dates = future.result()
            all_cross_dates[stock_code] = {
                "Golden_Cross": golden_dates,
                "Death_Cross": death_dates,
            }

    return all_cross_dates


# 带有重试机制的 yfinance 数据下载函数。
def download_with_retry(stock_code: str, max_retries=2, retry_delay=3):
    end_date = current_date
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_cdr_daily(
                symbol=stock_code,
                start_date="2015-01-01",
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            if not df.empty:  # 检查数据是否下载成功
                return df
            else:
                raise ValueError("Empty DataFrame returned.")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {stock_code}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # 等待后重试
                end_date = end_date - timedelta(1)
            else:
                print(f"Max retries reached for {stock_code}. Skipping...")
    return pd.DataFrame()  # 如果失败，返回空 DataFrame


# %%
file_name = "./txt_lib/stock_code.txt"
with open(file_name, "r") as file:
    stock_list = [line.strip() for line in file if line.strip()]

# 限制股票数量
stock_list = stock_list[:1000]

# all_cross_dates = process_stocks(stock_list)
all_cross_dates = common.Final_process_KDJ.process_stocks(stock_list)

# %%
try:
    folder = Path("./output/golden_output")
    folder.mkdir()
    folder = Path("./output/death_output")
    folder.mkdir()
    folder = Path("./output/操作1.1")
    folder.mkdir()
except Exception as e:
    pass

# 清理两个文件夹
common.Final_file.clean_folder("./output/death_output")
common.Final_file.clean_folder("./output/golden_output")
common.Final_file.clean_folder("./output/操作1.1")

# %%
# 选出今日金叉list
today_gold_cross_list = []
for stock_code in all_cross_dates.keys():
    if all_cross_dates[stock_code]["Golden_Cross"]:
        today_gold_cross_list.append(stock_code)

# 选出今日死叉list
today_death_cross_list = []
for stock_code in all_cross_dates.keys():
    if all_cross_dates[stock_code]["Death_Cross"]:
        today_death_cross_list.append(stock_code)

# KDJ日金 + MACD 2 条件 → 操作1.1
to_perfect_stock_list = common.Final_process.process_stocks(
    today_gold_cross_list, mode="MACD_2_condition"
)
output_file = "./output/操作1.1/操作1.1.xlsx"
common.Final_file.output_excel(to_perfect_stock_list, output_file, condition="完美")
common.Final_file.mk_pic(output_file)  # 生成所有图片

# %%
# 生成今日金叉 死叉Excel
common.Final_file_KDJ.output_excel(all_cross_dates)

# 生成今日金叉 死叉TXT
today_death_txt_path = "./output/death_output/today_death_cross.txt"
with open(today_death_txt_path, "w") as file:
    for item in today_death_cross_list:
        file.write(item + "\n")  # 每个元素后加换行符

today_gold_txt_path = "./output/golden_output/today_gold_cross.txt"
with open(today_gold_txt_path, "w") as file:
    for item in today_gold_cross_list:
        file.write(item + "\n")  # 每个元素后加换行符

# %%
output_file = "./output/golden_output/golden_cross_summary.xlsx"
common.Final_file.mk_pic(output_file)
output_file = "./output/death_output/death_cross_summary.xlsx"
common.Final_file.mk_pic(output_file)

# %%
10000/8/365



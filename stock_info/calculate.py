import akshare as ak
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import os

import common

class KDJ:

    @staticmethod
    def calculate_kdj(df: DataFrame, n=9):
        """计算 KDJ 指标"""
        low_min = df["low"].rolling(window=n).min()
        high_max = df["high"].rolling(window=n).max()
        rsv = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["K"] = round(rsv.ewm(com=2).mean(), 3)  # K线
        df["D"] = round(df["K"].ewm(com=2).mean(), 3)  # D线
        df["J"] = round(3 * df["K"] - 2 * df["D"], 3)  # J线
        return df

    @staticmethod
    def calculate_weekly_kdj(df: DataFrame):
        """按周重新采样,重新计算周K线数据的KDJ"""
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

        return KDJ.calculate_kdj(weekly_data)

    @staticmethod
    def find_kdj_golden_cross(df: DataFrame):
        """查找 KDJ 金叉和死叉，并判断当日股票涨跌"""
        df["Golden_Cross"] = (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))
        df["Dead_Cross"] = (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))

        # 判断股票涨跌：今日收盘价高于昨日收盘价则为上涨（Up），否则为下跌（Down）
        df["Price_Change"] = df["close"].diff()  # 计算收盘价变化
        df["Trend"] = df["Price_Change"].apply(lambda x: "Up" if x > 0 else "Down")

        return df

    @staticmethod
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

    @staticmethod
    def filter_golden_cross_without_dead_cross(data: DataFrame, periods: list):
        current_date = datetime.now() + timedelta(days=1)
        one_week_ago = current_date - timedelta(days=7)  # 计算一周之前的日期
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

    @staticmethod
    # 查找符合的股票
    def get_recent_golden_cross_dates(stock_code: str):
        current_date = datetime.now() + timedelta(days=1)
        one_week_ago = current_date - timedelta(days=7)  # 计算一周之前的日期

        data = common.Download_KDJ.download_with_retry(stock_code)

        if data.empty:  # 检查是否下载失败
            print(f"Failed to download data for {stock_code}. Skipping...")
            return []

        # 换成ak了，不需要去掉第二层的ticker
        # data.columns = data.columns.droplevel(1)  # 去掉第二层的 ticker，变为单层索引

        data = KDJ.calculate_kdj(data)  # 计算日 KDJ
        weekly_data = KDJ.calculate_weekly_kdj(data)  # 计算周 KDJ
        weekly_data = KDJ.find_kdj_golden_cross(weekly_data)  # 查找周 K线金叉
        data = KDJ.find_kdj_golden_cross(data)  # 查找日线 KDJ 金叉和死叉
        periods = KDJ.get_golden_to_dead_cross_periods(
            weekly_data
        )  # 提取周 KDJ 金叉到死叉的日期段

        # 获取符合条件的金叉日期
        filtered_golden_cross_dates = KDJ.filter_golden_cross_without_dead_cross(
            data, periods
        )

        # 将 filtered_golden_cross_dates 转换为 pandas 的 DatetimeIndex 以便筛选
        filtered_golden_cross_dates = pd.to_datetime(filtered_golden_cross_dates)

        # 筛选出最近一周内的日期
        recent_dates = [
            date.strftime("%Y-%m-%d")
            for date in filtered_golden_cross_dates
            if one_week_ago <= date <= current_date
        ]

        return recent_dates

    @staticmethod
    # 查找周kdj金叉 + 当日kdj死叉
    def filter_golden_cross_with_dead_cross(data: DataFrame, periods: list):
        current_date = datetime.now() + timedelta(days=1)

        filtered_dates = []
        if periods:
            # 检查今日是否出现死叉
            today_dead_cross = data["Dead_Cross"].iloc[-1]
            if today_dead_cross:
                filtered_dates.append(current_date)

        return filtered_dates

    @staticmethod
    # 查找符合的股票
    def get_recent_death_cross_dates(stock_code: str):
        current_date = datetime.now() + timedelta(days=2)
        one_week_ago = current_date - timedelta(days=7)  # 计算一周之前的日期
        data = common.Download_KDJ.download_with_retry(stock_code)

        if data.empty:  # 检查是否下载失败
            print(f"Failed to download data for {stock_code}. Skipping...")
            return []

        # 换成ak了，不需要去掉第二层的ticker
        # data.columns = data.columns.droplevel(1)  # 去掉第二层的 ticker，变为单层索引

        data = KDJ.calculate_kdj(data)  # 计算日 KDJ
        weekly_data = KDJ.calculate_weekly_kdj(data)  # 计算周 KDJ
        weekly_data = KDJ.find_kdj_golden_cross(weekly_data)  # 查找周 K线金叉
        data = KDJ.find_kdj_golden_cross(data)  # 查找日线 KDJ 金叉和死叉
        periods = KDJ.get_golden_to_dead_cross_periods(
            weekly_data
        )  # 提取周 KDJ 金叉到死叉的日期段

        # 获取符合条件的金叉日期
        filtered_golden_cross_dates = KDJ.filter_golden_cross_with_dead_cross(data, periods)

        # 将 filtered_golden_cross_dates 转换为 pandas 的 DatetimeIndex 以便筛选
        filtered_golden_cross_dates = pd.to_datetime(filtered_golden_cross_dates)

        # 筛选出最近一周内的日期
        recent_dates = [
            date.strftime("%Y-%m-%d")
            for date in filtered_golden_cross_dates
            if one_week_ago <= date <= current_date
        ]

        return recent_dates

    @staticmethod
    # 查看最近（即本周）周KDJ是否死叉
    def get_week_death(stock_code: str):
        data = common.Download_KDJ.download_with_retry(stock_code)
        week_data = KDJ.calculate_weekly_kdj(data)
        week_data_cross = KDJ.find_kdj_golden_cross(week_data)
        # 查看最近（即本周）周KDJ是否死叉
        if week_data_cross['Dead_Cross'].iloc[-1]:
            return stock_code

    @staticmethod
    # 查看今日 日KDJ是否死叉
    def get_day_death(stock_code: str):
        data = common.Download_KDJ.download_with_retry(stock_code)
        day_data = KDJ.calculate_kdj(data)
        daily_data_cross = KDJ.find_kdj_golden_cross(day_data)
        # 查看今日 日KDJ是否死叉
        if daily_data_cross["Dead_Cross"].iloc[-1]:
            return stock_code

    @staticmethod
    # 查看今日 日KDJ是否死叉
    def get_day_J_turn_around(stock_code: str):
        data = common.Download_KDJ.download_with_retry(stock_code)
        day_data = KDJ.calculate_kdj(data)
        daily_data_cross = KDJ.find_kdj_golden_cross(day_data)
        # 查看今日 日KDJ是否死叉
        condition1 = daily_data_cross["J"].iloc[-1] < daily_data_cross["J"].iloc[-2]
        condition2 = daily_data_cross["J"].iloc[-1] > daily_data_cross["K"].iloc[-1]
        if condition1 and condition2:
            return stock_code

    @staticmethod
    # 找出接近金叉的股票
    def get_near_golden_cross(stock_code: str):
        current_date = datetime.now()
        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="2015-01-01",
            end_date=current_date.strftime("%Y-%m-%d"),
            adjust="qfq",
        )  # qfq=前复权

        df = KDJ.calculate_kdj(df)

        condition1 = (-df["K"].iloc[-2] + df["D"].iloc[-2]) - (-df["K"].iloc[-1] + df["D"].iloc[-1]) > 3# 昨日KD差值大于今日
        condition2 = df["K"].iloc[-1] < df["D"].iloc[-1] # 今日K < D
        condition3 = ((df["D"].iloc[-1] - df["J"].iloc[-1]) < 12) # 今日D J 差值
        condition4 = (df["J"].iloc[-1] - df["J"].iloc[-2]) > 15 # 今昨 J 差值
        if (condition1 and condition2 and (condition3 or condition4)):
            return stock_code

    @staticmethod
    # 找出J < K
    def get_J_K(stock_code: str):
        current_date = datetime.now()
        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="2015-01-01",
            end_date=current_date.strftime("%Y-%m-%d"),
            adjust="qfq",
        )  # qfq=前复权

        df = KDJ.calculate_kdj(df)

        if df["J"].iloc[-1] < df["K"].iloc[-1]:
            return stock_code

class MACD:

    @staticmethod
    def calculate_ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    # 传统方法计算MACD
    def calculate_macd(df: DataFrame, short=12, long=26, signal=9):
        df["EMA_short"] = MACD.calculate_ema(df["close"], short)
        df["EMA_long"] = MACD.calculate_ema(df["close"], long)
        df["DIF"] = df["EMA_short"] - df["EMA_long"]
        df["DEA"] = MACD.calculate_ema(df["DIF"], signal)
        df["MACD_hist"] = 2 * (df["DIF"] - df["DEA"])

        # 计算金叉
        df["MACD_Golden"] = (df["DIF"].shift(1) <= df["DEA"].shift(1)) & (
            df["DIF"] > df["DEA"]
        )
        df["MACD_Death"] = (df["DIF"].shift(1) >= df["DEA"].shift(1)) & (
            df["DIF"] < df["DEA"]
        )
        return df

    # 将日线数据转换为月线数据
    @staticmethod
    def get_monthly_kline(df: DataFrame):
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # 添加年月列
        df["year_month"] = df["date"].dt.to_period("M")

        # 生成月线数据
        monthly_df = (
            df.groupby("year_month")
            .agg(
                {
                    "open": "first",  # 月开盘价取第一个交易日
                    "high": "max",  # 月最高价取最高点
                    "low": "min",  # 月最低价取最低点
                    "close": "last",  # 月收盘价取最后一个交易日
                    "volume": "sum",  # 月成交量取总和
                }
            )
            .reset_index()
        )

        # # 将year_month转换为datetime以便后续处理
        # monthly_df["date"] = monthly_df["year_month"].dt.to_timestamp()

        return monthly_df

    @staticmethod
    # 检查最近月diff是否刚上零线
    def check_monthly_macd_DIF(stock_code: str, df: DataFrame):
        # 转换为月线数据
        monthly_df = MACD.get_monthly_kline(df)

        # 计算月MACD
        monthly_df = MACD.calculate_macd(monthly_df)

        # 检查最近月DIF>0且上月DIF<0
        if monthly_df["DIF"].iloc[-1] > 0 and monthly_df["DIF"].iloc[-2] < 0:
            return stock_code

        return None

    @staticmethod
    # 获取股票的MACD DIF
    def get_macd(stock_code: str):
        try:
            # 下载股票日线数据
            data = common.Download.download_with_retry(stock_code)

            # 检查月DIF
            result = MACD.check_monthly_macd_DIF(stock_code, data)
            return result
        except Exception as e:
            print(f"处理股票 {stock_code} 时出错: {e}")
            return None

    @staticmethod
    # 找出日MACD小于0的
    def get_daily_macd_below0(stock_code: str):
        current_date = datetime.now()
        end_date = current_date
        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="2015-01-01",
            end_date=end_date.strftime("%Y-%m-%d"),
            adjust="qfq",
        )  # qfq=前复权

        daily_macd = MACD.calculate_macd(df)
        if daily_macd["MACD_hist"].iloc[-1]<0:
            return stock_code


class Mix:
    @staticmethod
    def get_MACD_2_condition(stock_code:str):
        '''找出满足macd两条件的股票'''
        data = common.Download.download_with_retry(stock_code)
        day_MACD_data = MACD.calculate_macd(data)

        # 查看DIF刚刚>0
        if day_MACD_data["DIF"].iloc[-1] > 0 and day_MACD_data["DIF"].iloc[-2] < 0:
            return stock_code

        # 零线上金叉
        elif (
            day_MACD_data["MACD_Golden"].iloc[-1] and day_MACD_data["DIF"].iloc[-1] > 0
        ):
            return stock_code

    @staticmethod
    def check_golden_cross_condition(df: DataFrame):
        """最近一次金叉发生在三周，且金叉发生后，不能发生死叉"""
        week_df = KDJ.calculate_weekly_kdj(df)
        week_cross = KDJ.find_kdj_golden_cross(week_df)

        # 找到最近一次 金叉 的索引
        golden_cross_indices = week_cross[week_cross["Golden_Cross"] == True].index

        # 获取最近一次 金叉 的索引
        latest_golden_cross_index = golden_cross_indices[-1]

        # 检查是否在最后三行内
        if latest_golden_cross_index not in week_cross.index[-3:]:
            return False  # 最近的 金叉 不在最后三行

        # 检查 金叉 之后是否有 死叉
        subsequent_rows = week_cross.loc[latest_golden_cross_index:]
        if subsequent_rows["Dead_Cross"].any():
            return False  # 金叉 之后有 死叉

        return True  # 满足所有条件

    @staticmethod
    def get_k_line_type(target_row: Series):
        """判断今日K线是阴线/阳线"""
        # 获取开盘价和收盘价
        open_price = target_row["open"]
        close_price = target_row["close"]

        # 判断K线类型
        if close_price > open_price:
            return "阳线"
        elif close_price < open_price:
            return "阴线"
        else:
            return "平线"

    @staticmethod
    def get_1st_neg_k_line(tod_row: Series, yest_row: Series):
        '''判断是否今日是第一根阴线'''
        tod_type = Mix.get_k_line_type(tod_row)
        yest_type = Mix.get_k_line_type(yest_row)

        if tod_type == "阴线" and yest_type == "阳线" or yest_type == "平线":
            return True

        return False

    @staticmethod
    def get_op1_1_cond3(stock_code: str):
        """判断是否今日是第一根阴线 + 周kdj金叉三周内"""
        current_date = datetime.now()

        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="2015-01-01",
            end_date=current_date.strftime("%Y-%m-%d"),
            adjust="qfq",
        )  # qfq=前复权

        tod_row = df.iloc[-1] # 今日数据
        yest_row = df.iloc[-2] # 昨日数据

        condition1 = Mix.get_1st_neg_k_line(tod_row, yest_row)
        condition2 = Mix.check_golden_cross_condition(df)

        if condition1 and condition2:
            return stock_code


class Bol:
    @staticmethod
    def calculate_bollinger_bands(df: DataFrame, window=26, num_std=2):
        """计算布林线"""
        df = df.sort_index(ascending=True)
        df["BOLL_MID"] = df["close"].rolling(window=window).mean()
        rolling_std = df["close"].rolling(window=window).std()
        df["BOLL_UPPER"] = df["BOLL_MID"] + (rolling_std * num_std)
        df["BOLL_LOWER"] = df["BOLL_MID"] - (rolling_std * num_std)
        return df

    @staticmethod
    def detect_bollinger_touch(df: DataFrame):
        """检测布林线碰轨情况"""
        touch_events = []

        for idx, row in df.iterrows():
            # 跳过NaN值（窗口期内无数据）
            if pd.isna(row["BOLL_UPPER"]) or pd.isna(row["BOLL_LOWER"]):
                continue

            # 检测触碰逻辑
            low_price = row["low"]
            touch_type = None

            # 检测上轨触碰（收盘价>=上轨）
            if low_price >= row["BOLL_UPPER"]:
                touch_type = "上轨"
            # 检测下轨触碰（收盘价<=下轨）
            elif low_price <= row["BOLL_LOWER"] + 0.02:
                touch_type = "下轨"
            # 检测中轨触碰（收盘价与中轨差异<0.5%）
            elif abs(low_price - row["BOLL_MID"]) / row["BOLL_MID"] < 0.005:
                touch_type = "中轨"

            if touch_type:
                touch_time = (
                    idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
                )
                touch_events.append([touch_time, touch_type])

        return touch_events

    @staticmethod
    def get_bol(stock_code:str, sub_mode = "中下"):
        current_date = datetime.now()
        end_date = current_date

        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="2010-01-01",
            end_date=end_date.strftime("%Y-%m-%d"),
            adjust="qfq",
        )

        # # 预处理数据
        # df.index = pd.to_datetime(df["date"])
        # df = df.sort_index()
        df = Bol.calculate_bollinger_bands(df)
        touch_records = Bol.detect_bollinger_touch(df)  # 获取碰轨事件

        # op2 = []  # 移至操作2

        # current_date = "2025-05-16"
        current_date = datetime.now().strftime("%Y-%m-%d")  # today
        last_touch_date = touch_records[-1][0] # 最近一次碰轨日期
        if last_touch_date == current_date:
            last_touch_type = touch_records[-1][1]  # 本回轨道
            bef_last_touch_type = touch_records[-2][1]  # 上回轨道
            bef_last_touch_date = touch_records[-2][0]  # 最近二次碰轨日期

            if sub_mode == "中下":

                if (
                    last_touch_date != bef_last_touch_date
                    and (last_touch_type == "中轨" or last_touch_type == "下轨")
                    and bef_last_touch_type == "上轨"
                ):
                    return stock_code
            
            elif sub_mode == "下":
                if (
                    last_touch_date != bef_last_touch_date
                    and (last_touch_type == "下轨")
                    and bef_last_touch_type == "上轨"
                ):
                    return stock_code

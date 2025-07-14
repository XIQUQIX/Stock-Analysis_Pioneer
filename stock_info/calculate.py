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
import pickle
import gc
import copy

import common

# 全局加载 pickle 文件
with open("./txt_lib/daily_df.pkl", "rb") as f:
    GLOBAL_STOCK_DATA = pickle.load(f)


class Reshape_data:
    @staticmethod
    def get_week_df(df: DataFrame):
        """将日线数据转换为周线数据"""
        df["date"] = pd.to_datetime(df["date"])  # 确保 date 列是 datetime 格式
        df = df.set_index("date")  # 把 date 设为索引

        weekly_data = (
            df.resample("W-FRI")  # 取当周五的数据
            .agg(
                {
                    "open": "first",  # 周开盘价取第一个交易日
                    "high": "max",  # 周最高价取最高点
                    "low": "min",  # 周最低价取最低点
                    "close": "last",  # 周收盘价取最后一个交易日
                    "volume": "sum",  # 周成交量取总和
                }
            )
            .dropna()
        )

        return weekly_data

    @staticmethod
    def get_month_df(df: DataFrame):
        """将日线数据转换为月线数据"""
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

        return monthly_df


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
        df["Golden_Cross"] = (df["K"] > df["D"]) & (
            df["K"].shift(1) <= df["D"].shift(1)
        )
        df["Dead_Cross"] = (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))

        # 判断股票涨跌：今日收盘价高于昨日收盘价则为上涨（Up），否则为下跌（Down）
        df["Price_Change"] = df["close"].diff()  # 计算收盘价变化
        df["Trend"] = df["Price_Change"].apply(lambda x: "Up" if x > 0 else "Down")

        return df

    @staticmethod
    def get_golden_to_dead_cross_periods(df: DataFrame):
        """查找最近金叉时段 且 必须是1最近一周"""
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
        """日金叉 + 今日股价上涨"""
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

        # data = common.Download_KDJ.download_with_retry(stock_code)

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

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
        # data = common.Download_KDJ.download_with_retry(stock_code)

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

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
        filtered_golden_cross_dates = KDJ.filter_golden_cross_with_dead_cross(
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
    def get_week_death(stock_code: str):
        """
        查看周KDJ是否处于死叉 == 本周D值大于J值
        """
        # data = common.Download_KDJ.download_with_retry(stock_code)

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

        week_data = KDJ.calculate_weekly_kdj(data)

        if week_data["D"].iloc[-1] > week_data["J"].iloc[-1]:
            return stock_code

    @staticmethod
    def get_day_death(stock_code: str):
        """查看今日 日KDJ是否处于死叉 == 当日D值大于J值"""
        # data = common.Download_KDJ.download_with_retry(stock_code)

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

        day_data = KDJ.calculate_kdj(data)

        # 查看今日 日KDJ情况
        if day_data["D"].iloc[-1] > day_data["J"].iloc[-1]:
            return stock_code

    @staticmethod
    def get_day_J_turn_around(stock_code: str):
        """查看J线是否拐头"""
        # data = common.Download_KDJ.download_with_retry(stock_code)

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

        day_data = KDJ.calculate_kdj(data)
        daily_data_cross = KDJ.find_kdj_golden_cross(day_data)
        # J线拐头 + J线在K线之上
        condition1 = daily_data_cross["J"].iloc[-1] < daily_data_cross["J"].iloc[-2]
        condition2 = daily_data_cross["J"].iloc[-1] > daily_data_cross["K"].iloc[-1]
        if condition1 and condition2:
            return stock_code

    @staticmethod
    def get_day_golden(stock_code: str):
        """
        查看今日 日KDJ是否处在金叉阶段
        J > K
        """

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

        day_data = KDJ.calculate_kdj(data)
        # 查看今日 日KDJ情况
        if day_data["K"].iloc[-1] < day_data["J"].iloc[-1]:
            return stock_code

    @staticmethod
    def get_near_golden_cross(stock_code: str):
        # """接近金叉的股票 + DIF > 0 + MACD > 0"""

        # data = GLOBAL_STOCK_DATA
        # df = pd.DataFrame(data[stock_code])

        # df = KDJ.calculate_kdj(df)
        # df = MACD.calculate_macd(df)

        # condition1 = (-df["K"].iloc[-2] + df["D"].iloc[-2]) - (
        #     -df["K"].iloc[-1] + df["D"].iloc[-1]
        # ) > 3  # 昨日KD差值大于今日
        # condition2 = df["K"].iloc[-1] < df["D"].iloc[-1]  # 今日K < D
        # condition3 = (df["D"].iloc[-1] - df["J"].iloc[-1]) < 12  # 今日D J 差值
        # condition4 = (df["J"].iloc[-1] - df["J"].iloc[-2]) > 15  # 今昨 J 差值

        # condition5 = (df["MACD_hist"].iloc[-1] > 0) and (
        #     df["DIF"].iloc[-1] > 0
        # )  # 今日DIF > 0 + MACD > 0

        # if (condition1 and condition2 and (condition3 or condition4)) and condition5:
        #     return stock_code
        """
        1. (今日收盘价 - 中轨) / 收盘价 < 4% or (今日收盘价 - 下轨) / 收盘价 < 4%

        2. J今日- J昨日 > 0 + 第一次

        3. 周bol 中轨 本周 >= 上周

        4. DIF > 0 + MACD > 0
        """
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        week_df = Reshape_data.get_week_df(df)
        bol_df = Bol.calculate_bollinger_bands(df)

        df = KDJ.calculate_kdj(df)
        df = MACD.calculate_macd(df)

        today_close = df["close"].iloc[-1]
        today_mid = bol_df["BOLL_MID"].iloc[-1]
        today_low = bol_df["BOLL_LOWER"].iloc[-1]

        # (今日收盘价 - 中轨) / 收盘价 < 4% or (今日收盘价 - 下轨) / 收盘价 < 4%
        condition1 = ((abs(today_close - today_mid) / today_close) < 0.04) or (
            (abs(today_close - today_low) / today_close) < 0.04
        )

        # J今日 > J昨日  + 第一次
        condition2 = (df["J"].iloc[-1] > df["J"].iloc[-2]) and (
            df["J"].iloc[-3] > df["J"].iloc[-2]
        )

        # 周bol 中轨 本周 >= 上周
        week_bol_df = Bol.calculate_bollinger_bands(week_df)
        condition3 = round(week_bol_df["BOLL_MID"].iloc[-1], 3) >= round(
            week_bol_df["BOLL_MID"].iloc[-2], 3
        )# 近似到小数点后三位

        # 今日DIF > 0 + MACD > 0
        condition4 = (df["MACD_hist"].iloc[-1] > 0) and (df["DIF"].iloc[-1] > 0)
        
        if condition1 and condition2 and condition3 and condition4:
            return stock_code

    @staticmethod
    def get_J_K(stock_code: str):
        """找出J < K"""
        # df = common.Download.download_with_retry(stock_code)

        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        df = KDJ.calculate_kdj(df)

        if df["J"].iloc[-1] < df["K"].iloc[-1]:
            return stock_code

    @staticmethod
    def get_month_kdj_golden(stock_code: str):
        """找出当月 月KDJ金叉 的股票"""
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        month_df = MACD.get_month_df(df)

        month_df = KDJ.calculate_kdj(month_df)
        month_df = KDJ.find_kdj_golden_cross(month_df)

        # 当月 月kdj发生金叉
        if month_df["Golden_Cross"].iloc[-1]:
            return stock_code


class MACD:

    @staticmethod
    def calculate_ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_macd(df: DataFrame, short=12, long=26, signal=9):
        """传统方法计算MACD"""
        df["EMA_short"] = round(MACD.calculate_ema(df["close"], short), 3)
        df["EMA_long"] = round(MACD.calculate_ema(df["close"], long), 3)
        df["DIF"] = round(df["EMA_short"] - df["EMA_long"], 3)
        df["DEA"] = round(MACD.calculate_ema(df["DIF"], signal), 3)
        df["MACD_hist"] = round(2 * (df["DIF"] - df["DEA"]), 3)

        # 计算金叉
        df["MACD_Golden"] = (df["DIF"].shift(1) <= df["DEA"].shift(1)) & (
            df["DIF"] > df["DEA"]
        )
        df["MACD_Death"] = (df["DIF"].shift(1) >= df["DEA"].shift(1)) & (
            df["DIF"] < df["DEA"]
        )
        return df

    @staticmethod
    def get_month_df(df: DataFrame):
        """将日线数据转换为月线数据"""
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
    def check_monthly_macd_DIF(stock_code: str):
        """检查月diff是否刚上零线"""
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        # 转换为月线数据
        monthly_df = MACD.get_month_df(df)

        # 计算月MACD
        monthly_df = MACD.calculate_macd(monthly_df)

        # 检查最近月DIF>0且上月DIF<0
        if monthly_df["DIF"].iloc[-1] > 0 and monthly_df["DIF"].iloc[-2] < 0:
            return stock_code

        del df
        del monthly_df
        gc.collect()

        return None

    @staticmethod
    def get_daily_macd_below0(stock_code: str):
        """找出日MACD小于0的"""
        # df = common.Download.download_with_retry(stock_code)

        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        daily_macd = MACD.calculate_macd(df)
        if daily_macd["MACD_hist"].iloc[-1] < 0:
            return stock_code


class Mix:
    @staticmethod
    def get_MACD_2_condition(stock_code: str):
        """找出满足macd两条件的股票"""
        # data = common.Download.download_with_retry(stock_code)

        stock_data = GLOBAL_STOCK_DATA
        data = pd.DataFrame(stock_data[stock_code])

        day_MACD_data = MACD.calculate_macd(data)

        # 查看DIF>0   (条件1)
        condition1 = day_MACD_data["DIF"].iloc[-1] > 0 

        # 今日不能碰上轨   (条件3)
        df = Bol.calculate_bollinger_bands(data)
        touch_records = Bol.detect_bollinger_touch(df)  # 获取碰轨事件

        current_date = datetime.now().strftime("%Y-%m-%d")  # today
        last_touch_date = touch_records[-1][0]  # 最近一次碰轨日期
        last_touch_type = touch_records[-1][1]  # 本回轨道

        # 今日碰上轨
        condition3 = (last_touch_date == current_date) and last_touch_type == "上轨"

        if (condition1) and (not condition3):
            return stock_code

    @staticmethod
    def check_golden_cross_condition(df: DataFrame):
        """最近一次金叉发生在两周,且金叉发生后,不能发生死叉"""
        week_df = KDJ.calculate_weekly_kdj(df)
        week_cross = KDJ.find_kdj_golden_cross(week_df)

        # 找到最近一次 金叉 的索引
        golden_cross_indices = week_cross[week_cross["Golden_Cross"] == True].index

        # 获取最近一次 金叉 的索引
        latest_golden_cross_index = golden_cross_indices[-1]

        # 检查是否在最后三行内
        if latest_golden_cross_index not in week_cross.index[-2:]:
            return False  # 最近的 金叉 不在最后两行

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
        """判断是否今日是第一根阴线"""
        tod_type = Mix.get_k_line_type(tod_row)
        yest_type = Mix.get_k_line_type(yest_row)

        if tod_type == "阴线" and yest_type == "阳线" or yest_type == "平线":
            return True

        return False

    @staticmethod
    def get_2nd_neg_k_line(tod_row: Series, yest_row: Series, prev_row: Series):
        """判断是否今日是第二根阴线"""
        tod_type = Mix.get_k_line_type(tod_row)
        yest_type = Mix.get_k_line_type(yest_row)
        prev_type = Mix.get_k_line_type(prev_row)

        if tod_type == "阴线" and yest_type == "阴线" and prev_type != "阴线":
            return True

        return False

    @staticmethod
    def get_op1_1_2cond(stock_code: str):
        # """
        # 判断是否今日是第一根阴线 + 周kdj金叉三周内
        # + 昨日阳线碰Bol上轨 + 今日最高价小于昨日最高价
        # """
        # # df = common.Download.download_with_retry(stock_code)

        # data = GLOBAL_STOCK_DATA
        # df = pd.DataFrame(data[stock_code])

        # tod_row = df.iloc[-1]  # 今日数据
        # yest_row = df.iloc[-2]  # 昨日数据

        # condition1 = Mix.get_1st_neg_k_line(tod_row, yest_row)  # 今日是第一根阴线
        # condition2 = Mix.check_golden_cross_condition(df)  # 周kdj金叉三周内

        # # 预处理数据
        # df = Bol.calculate_bollinger_bands(df)
        # yest_date = (datetime.now() - timedelta(days=1)).strftime(
        #     "%Y-%m-%d"
        # )  # yesterday
        # touch_records = Bol.detect_bollinger_touch(df)  # 获取碰轨事件
        # last_touch_date = touch_records[-1][0]  # 最近一次碰轨日期
        # last_touch_type = touch_records[-1][1]  # 本回轨道
        # bef_last_touch_type = touch_records[-2][1]  # 上回轨道
        # bef_last_touch_date = touch_records[-2][0]  # 最近二次碰轨日期

        # condition3_1 = (last_touch_date == yest_date) and (
        #     last_touch_type == "上轨"
        # )  # 倒数第一次碰轨情况 (今日未碰轨)
        # condition3_2 = (bef_last_touch_date == yest_date) and (
        #     bef_last_touch_type == "上轨"
        # )  # 倒数第二次碰轨情况 (今日也碰轨)

        # condition3 = condition3_1 or condition3_2  # 昨日/今日 碰上轨

        # tod_high = tod_row["high"]  # 今日最高价
        # yest_high = yest_row["high"]  # 昨日最高价

        # condition4 = tod_high < yest_high  # 今日最高价小于昨日最高价

        # if condition1 and condition2 and condition3 and condition4:
        #     return stock_code
        """
        今日是阴线 + abs(最低价 - 中轨) / 收盘价 < 3% or abs(最低价 -下轨) / 收盘价 < 3%
        abs(上轨 - 收盘价) / 收盘价 > 10%  + 周kdj金叉2周内
        两者符合其一
        """
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        day_MACD_data = MACD.calculate_macd(df)

        # 查看DIF>0   (条件new)
        condition_new = day_MACD_data["DIF"].iloc[-1] > 0 

        tod_row = df.iloc[-1]  # 今日数据
        tod_type = Mix.get_k_line_type(tod_row)

        today_low = df["low"].iloc[-1]
        today_close = df["close"].iloc[-1]
        bol_df = Bol.calculate_bollinger_bands(df)
        bol_upper = round(bol_df["BOLL_UPPER"].iloc[-1], 3)
        bol_mid = round(bol_df["BOLL_MID"].iloc[-1],3)
        bol_low = round(bol_df["BOLL_LOWER"].iloc[-1], 3)

        # 条件1
        condition1_1 = (tod_type == "阴线")

        condition1_2 = (abs(today_low - bol_mid) / today_close < 0.03) or (
            abs(today_low - bol_low) / today_close < 0.03
        )

        condition1 = (condition1_1 and condition1_2)

        # 条件2
        condition2_1 = (abs(bol_upper - today_close) / today_close) > 0.1
        condition2_2 = Mix.check_golden_cross_condition(df)
        condition2 = (condition2_1 and condition2_2)
        
        if (condition1 or condition2) and condition_new:
            return stock_code

    @staticmethod
    def is_third_last_high_max(df: DataFrame, column="high"):
        """检查df中列倒数第3行的值是否是倒数10行中的最大值"""
        last_n_rows = df[column].tail(10)  # 获取 最近10天 数据
        target_value = df[column].iloc[-3]  # 获取 '第三日' 数据
        is_max = target_value == last_n_rows.max()  # 检查是否最大值

        return is_max

    @staticmethod
    def is_touch_mid(df: DataFrame):
        """最近两天都不能碰中轨"""
        # 预处理数据
        df = Bol.calculate_bollinger_bands(df)
        touch_records = Bol.detect_bollinger_touch(df)  # 获取碰轨事件
        current_date = datetime.now().strftime("%Y-%m-%d")  # today
        one_day_bef = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        last_touch_date = touch_records[-1][0]  # 最近一次碰轨日期
        last2_touch_date = touch_records[-2][0]  # 最近二次碰轨日期
        last_touch_type = touch_records[-1][1]  # 最近一次轨道
        last2_touch_type = touch_records[-2][1]  # 最近二次轨道

        if (last_touch_date == current_date and last_touch_type == "中轨") or (
            last2_touch_date == one_day_bef and last2_touch_type == "中轨"
        ):
            return False

        return True

    @staticmethod
    def get_op2_cond3(stock_code: str):
        """判断今日是第二根阴线 + 日MACD > 0 + '第三日'创新高 + 最近两天不能碰中轨"""
        # df = common.Download.download_with_retry(stock_code)

        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        tod_row = df.iloc[-1]  # 今日数据
        yest_row = df.iloc[-2]  # 昨日数据
        prev_row = df.iloc[-3]  # 前日数据

        condition1 = Mix.get_2nd_neg_k_line(tod_row, yest_row, prev_row)  # 第二根阴线

        day_macd = MACD.calculate_macd(df)
        condition2 = day_macd["MACD_hist"].iloc[-1] > 0  # 日MACD > 0

        condition3 = Mix.is_third_last_high_max(df)  #'第三日'创新高
        condition4 = Mix.is_touch_mid(df)  # 最近两天不能碰中轨

        if condition1 and condition2 and condition3 and condition4:
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
            high_price = row["high"]
            close_price = row["close"]
            touch_type = None

            # 检测上轨触碰（收盘最高价>=上轨）
            if high_price >= row["BOLL_UPPER"]:
                touch_type = "上轨"
            # 检测下轨触碰（收盘最低价<=下轨）(将下轨抬高3分)
            elif low_price <= row["BOLL_LOWER"] + 0.03:
                touch_type = "下轨"
            # 检测中轨触碰（收盘价与中轨差异<0.5%）
            elif abs(low_price - row["BOLL_MID"]) / row["BOLL_MID"] < 0.005:
                touch_type = "中轨"

            if touch_type:
                touch_time = row["date"].strftime("%Y-%m-%d")
                # touch_time = (
                #     idx.strftime("%Y-%m-%d")
                #     if isinstance(idx, pd.Timestamp)
                #     else str(idx)
                # )

                touch_events.append([touch_time, touch_type])

        return touch_events

    @staticmethod
    def get_bol(stock_code: str, sub_mode="中下"):
        """
        股票前次碰轨必须是上轨
        不同sub_mode选择本次碰轨情况， 默认中轨or下轨，附加 昨日最高价 > 今日最高价
        碰下轨无此附加条件
        """
        # df = common.Download.download_with_retry(stock_code)

        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        # 预处理数据
        df = Bol.calculate_bollinger_bands(df)
        touch_records = Bol.detect_bollinger_touch(df)  # 获取碰轨事件

        # current_date = "2025-05-23"
        current_date = datetime.now().strftime("%Y-%m-%d")  # today
        last_touch_date = touch_records[-1][0]  # 最近一次碰轨日期
        if last_touch_date == current_date:
            last_touch_type = touch_records[-1][1]  # 本回轨道
            bef_last_touch_type = touch_records[-2][1]  # 上回轨道
            bef_last_touch_date = touch_records[-2][0]  # 最近二次碰轨日期

            if sub_mode == "中下":

                if (
                    last_touch_date != bef_last_touch_date
                    and (last_touch_type == "中轨" or last_touch_type == "下轨")
                    and bef_last_touch_type == "上轨"
                    and df["high"].iloc[-2]
                    > df["high"].iloc[-1]  # 昨日最高价 > 今日最高价
                ):
                    return stock_code

            elif sub_mode == "下":
                if (
                    last_touch_date != bef_last_touch_date
                    and (last_touch_type == "下轨")
                    and bef_last_touch_type == "上轨"
                ):
                    return stock_code

    @staticmethod
    def get_mid_up(stock_code: str):
        """找出中线向上的股票"""
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        month_df = Reshape_data.get_month_df(df)

        # 预处理数据
        month_bol_df = Bol.calculate_bollinger_bands(month_df)

        # 中线向上
        condition1 = month_bol_df["BOLL_MID"].iloc[-2] <= month_bol_df["BOLL_MID"].iloc[-1]

        del df, month_bol_df
        gc.collect()

        if condition1:
            return stock_code

    @staticmethod
    def get_month_mid_up(stock_code: str):
        """找出中线向上的股票"""
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        month_df = MACD.get_month_df(df)

        # 预处理数据
        df = Bol.calculate_bollinger_bands(month_df)

        # 中线向上
        condition1 = df["BOLL_MID"].iloc[-2] <= df["BOLL_MID"].iloc[-1]

        del df
        gc.collect()

        if condition1:
            return stock_code

    @staticmethod
    def get_not_touch_upper(stock_code: str):
        """找出 当日没有碰上轨 的股票"""
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])

        # 预处理数据
        df = Bol.calculate_bollinger_bands(df)
        touch_records = Bol.detect_bollinger_touch(df)  # 获取碰轨事件

        current_date = datetime.now().strftime("%Y-%m-%d")  # today
        last_touch_date = touch_records[-1][0]  # 最近一次碰轨日期
        last_touch_type = touch_records[-1][1]  # 本回轨道

        del df
        del touch_records
        gc.collect()

        # 当日 + 碰上轨 == 不要
        if not (current_date == last_touch_date and last_touch_type == "上轨"):
            return stock_code


class Nine:
    def __init__(self) -> None:
        with open("./txt_lib/daily_df.pkl", "rb") as f:
            self.data = pickle.load(f)

    @staticmethod
    def find_decreasing_sequence(low_list: list[float], k=7):
        """
        从输入的列表 low_list 中选择一个长度为 k 的严格递减子序列，
        并确保子序列包含列表的最后一个元素
        返回: 选择的元素的index
        """
        n = len(low_list)
        if n < k:
            return []  # 列表长度不足

        if low_list[0] != max(low_list):
            return []

        res = []
        current = float("inf")  # 当前允许的最大值
        i = 0  # 当前遍历到的列表索引

        while i < n and len(res) < k:
            if low_list[i] < current:
                current = low_list[i]
                res.append(i)

            i += 1

        # 最后一个元素必须包含 + 必须找到k个元素
        if (n - 1 not in res) or len(res) != k:
            return []

        return res

    @staticmethod
    def get_nine(stock_code: str):
        """找出正好7天最低价递减的股票(可以不连续)"""
        try:
            # 获取股票日线数据
            # # akshare 新浪
            # df = common.Download.download_with_retry(stock_code)

            data = GLOBAL_STOCK_DATA
            df = pd.DataFrame(data[stock_code])

        except Exception as e:
            # 跳过数据获取失败的股票
            print(f"股票 {stock_code} 数据获取失败: {e}")
            return

        if df is None:
            return

        for days in range(7, 30):
            # 提取最近 days 天的最低价
            recent_row = df.tail(days)
            recent_lows = df["low"].tail(days).tolist()

            condition1 = Nine.find_decreasing_sequence(recent_lows, 7)  # 7天最低价递减

            if condition1:
                for pre_days in range(days, 30):
                    pre_lows = df["low"].tail(pre_days).tolist()
                    condition2 = Nine.find_decreasing_sequence(
                        pre_lows, 8
                    )  # 8天最低价递减
                    condition3 = Nine.find_decreasing_sequence(
                        pre_lows, 9
                    )  # 9天最低价递减
                    condition4 = Nine.find_decreasing_sequence(
                        pre_lows, 10
                    )  # 10天最低价递减
                    if condition2 or condition3 or condition4:
                        del df
                        gc.collect()
                        return

                # 获取下降日期
                dec_date = []
                for index in condition1:
                    date = recent_row["date"].iloc[index]
                    dec_date.append(date.strftime("%m-%d"))

                # 检查下降周期首日，最低价 & 最高价都是周期内最高价
                res_low = df["low"].tail(days + 1).tolist()  # 低价list 多找一天
                res_high = df["high"].tail(days + 1).tolist()  # 高价list 多找一天

                condition5 = (
                    res_low[1] == max(res_low[1:])  # 应该不需要
                    and res_high[1] == max(res_high)  # 向前多看一天
                    or (
                        res_low[0] > res_low[2] and res_high[0] > res_high[1]
                    )  # 前天可以代替开始第一日
                )

                if condition5:
                    del df
                    gc.collect()
                    return [stock_code, dec_date]

    @staticmethod
    def get_today(stock_code:str):
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        week_df = Reshape_data.get_week_df(df)
        month_df = Reshape_data.get_month_df(df)

        def get_4_condition(stock_code:str):
            """
            1 周KDJ 不死叉
            2 周macd dif > 0
            3 月kdj 不死叉
            4 月macd dif > 0
            """
            # condition1 (J > D)
            week_kdj_df = KDJ.calculate_kdj(week_df)
            condition1 = week_kdj_df["D"].iloc[-1] < week_kdj_df["J"].iloc[-1]

            # condition2
            week_macd_df = MACD.calculate_macd(week_df)
            condition2 = week_macd_df["DIF"].iloc[-1] > 0

            # condition3
            month_kdj_df = KDJ.calculate_kdj(month_df)
            condition3 = month_kdj_df["D"].iloc[-1] < month_kdj_df["J"].iloc[-1]

            # condition4
            month_macd_df = MACD.calculate_macd(month_df)
            condition4 = month_macd_df["DIF"].iloc[-1] > 0

            if condition1 or condition2 or condition3 or condition4:
                return True
            return False

        # 4条件
        condition1 = get_4_condition(stock_code)

        # 中轨向上 == 今日中轨大于昨日
        res_code = Bol.get_mid_up(stock_code)
        if res_code:
            condition2 = True
        else:
            condition2 = False

        # 昨日最高价 > 今日最高价
        condition3 = df["high"].iloc[-2] > df["high"].iloc[-1]

        if condition1 and condition2 and condition3:
            return stock_code

    @staticmethod
    def get_net_profit(stock_code: str):
        """获取股票净利润"""
        financial_abstract = ak.stock_financial_abstract_ths(symbol=stock_code[2:])
        # 净利润 为 str ("7.55亿")————去除最后一位,转换成float
        net_profit = float(financial_abstract["净利润"].iloc[0][:-1])

        return net_profit

    @staticmethod
    def check_net_profit_macd(stock_code: str):
        """
        利润 > 0 -> 保留
        利润 < 0 + 周&月macd dif > 0 -> 保留
        """
        # 净利润
        net_profit = Nine.get_net_profit(stock_code)

        if net_profit > 0:
            return stock_code
        else:
            data = GLOBAL_STOCK_DATA
            df = pd.DataFrame(data[stock_code])
            week_df = Reshape_data.get_week_df(df)
            month_df = Reshape_data.get_month_df(df)

            week_macd_df = MACD.calculate_macd(week_df)
            month_macd_df = MACD.calculate_macd(month_df)

            del df, week_df, month_df
            gc.collect()

            # 周 + 月macd DIF 都要大于0
            if week_macd_df["DIF"].iloc[-1] > 0 and month_macd_df["DIF"].iloc[-1] > 0:
                return stock_code


class Pioneer:

    @staticmethod
    def get_3_min_max_down(stock_code: str):
        """
        连续三周最低价下降 + (连续三周最高价下降 + 上周最高价大于周一最高价) + 连续三周成交量下降
        上周为week1, 上上周week2, 上上上周week3
        """
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        week_df = Reshape_data.get_week_df(df)

        # 周最低价
        week_1_low = week_df["low"].iloc[-2]
        week_2_low = week_df["low"].iloc[-3]
        week_3_low = week_df["low"].iloc[-4]

        # 周最高价
        cur_week_high = week_df["high"].iloc[-1]
        week_1_high = week_df["high"].iloc[-2]
        week_2_high = week_df["high"].iloc[-3]
        week_3_high = week_df["high"].iloc[-4]

        # 周成交量
        week_1_vol = week_df["volume"].iloc[-2]
        week_2_vol = week_df["volume"].iloc[-3]
        week_3_vol = week_df["volume"].iloc[-4]

        del df, week_df
        gc.collect()

        # 连续三周最低价下降
        condition1 = (week_3_low > week_2_low) and (week_2_low > week_1_low)

        # 连续三周最高价下降 + 上周最高价大于周一最高价
        condition2 = (week_3_high > week_2_high) and (week_2_high > week_1_high) and (week_1_high > cur_week_high)

        # 连续三周成交量下降
        condition3 = (week_3_vol > week_2_vol) and (week_2_vol > week_1_vol)

        if condition1 and condition2 and condition3:
            return True
        return False

    @staticmethod
    def get_1_percent(stock_code: str):
        """
        上周 abs(开盘-收盘)/(min(开盘，收盘)) < 1%
        上周为week1,上上周week2,上上上周week3
        """
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        week_df = Reshape_data.get_week_df(df)

        condition1 = (
            abs(week_df["open"].iloc[-2] - week_df["close"].iloc[-2])
            / min(week_df["open"].iloc[-2], week_df["close"].iloc[-2])
            < 0.01
        )

        del df, week_df
        gc.collect()

        if condition1:
            return True
        return False

    @staticmethod
    def get_4_5_rate(stock_code: str):
        """
        [第一周 abs(开盘-收盘)] * 4.5 < [min(第二周 abs(开盘-收盘),第三周 abs(开盘-收盘))]
        上周为week1,上上周week2,上上上周week3
        """
        data = GLOBAL_STOCK_DATA
        df = pd.DataFrame(data[stock_code])
        week_df = Reshape_data.get_week_df(df)

        l = 4.5 * abs(week_df["open"].iloc[-2] - week_df["close"].iloc[-2])
        r = min(
            abs(week_df["open"].iloc[-3] - week_df["close"].iloc[-3]),
            abs(week_df["open"].iloc[-4] - week_df["close"].iloc[-4]),
        )

        del df, week_df
        gc.collect()

        if l < r:
            return True
        return False

    @staticmethod
    def pioneer_1(stock_code: str):
        condition1 = Pioneer.get_3_min_max_down(stock_code)
        condition2 = Pioneer.get_1_percent(stock_code)
        condition3 = Pioneer.get_4_5_rate(stock_code)

        if condition1 and condition2 and condition3:
            return stock_code

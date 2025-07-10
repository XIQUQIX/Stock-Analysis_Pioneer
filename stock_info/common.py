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
import pickle
import gc
import psutil
import copy

import calculate

class Initialization:
    @staticmethod
    # 生成股票代码 名称 词典
    def generate_stock_dict(filename = "./txt_lib/stock_name.txt"):
        stock_dict = {}

        with open(filename, "r", encoding="utf-8") as file:
            # 读取文件内容
            content = file.read()

            # 按照股票名和股票代码的格式分割
            stock_info = content.split(")")  # 按照')'分割

            for info in stock_info:
                if "(" in info:  # 如果包含'('，说明这是一个有效的股票信息
                    code_name = info.split("(")  # 通过'('分割代码和名称
                    if len(code_name) == 2:
                        code = code_name[1]
                        name = code_name[0].strip()
                        stock_dict[code] = name

        return stock_dict
    
    @staticmethod
    def add_capital(stock_code: str):
        '''给六位数字股票代码加上sh/sz'''
        if stock_code.startswith("6"):
            processed_code = f"sh{stock_code}"
        else:
            processed_code = f"sz{stock_code}"
            
        return processed_code


class Read_pickle:
    @staticmethod
    def read_pickle_data(stock_code:str):
        """打开 daily_df.pkl 并返回所选股票的df"""
        with open("./txt_lib/daily_df.pkl", "rb") as f:
            data = pickle.load(f)

        df = pd.DataFrame(data[stock_code])
        return df

class Final_file:
    @staticmethod
    def output_excel(stock_list: list[str], output_file: str, condition="金叉"):
        """
        输出最终excel文件
        可以更改表头，默认是“金叉”
        """
        filename = "./txt_lib/stock_name.txt"  # 替换新文件路径
        stock_dict = Initialization.generate_stock_dict(filename)
        num = len(stock_list)
        data = []

        # 遍历所有股票金叉数据
        for stock_code in stock_list:
            cur_stock_code = str(stock_code[-6:])  # 去掉市场后缀，保留6位股票代码
            stock_name = stock_dict.get(cur_stock_code, "未知")  # 获取股票名称
            data.append({"股票代码": cur_stock_code, "股票名称": stock_name})

        df = pd.DataFrame(data)
        summary_row = pd.DataFrame(
            {"股票代码": [f"总计{condition}股票个数：{num}"], "股票名称": [""]}
        )  # 添加总计行到 DataFrame 的顶部

        df = pd.concat([summary_row, df], ignore_index=True)

        # output_file = ".\golden_cross_summary1.xlsx"
        df.to_excel(output_file, index=False, sheet_name=f"{condition}统计")
        print(f"Excel 文件已生成: {output_file}")

    @staticmethod
    def clean_folder(folder_path: str):
        """删除指定文件夹中的所有文件"""
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    @staticmethod
    def mk_pic(output_file: str):
        # 读取 Excel
        file_path = output_file
        df = pd.read_excel(file_path, usecols=[0])  # 只读取第一列
        directory = os.path.dirname(output_file)  # 提取目录部分

        # 解决中文乱码
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # 设置每张图最多显示的行数
        rows_per_image = 50

        # 按每 50 行划分数据并生成图片
        for i in range(0, len(df), rows_per_image):
            subset = df.iloc[i : i + rows_per_image]  # 取出 50 行数据
            fig, ax = plt.subplots(figsize=(5, len(subset) * 0.4))  # 调整图片大小
            ax.axis("tight")
            ax.axis("off")
            ax.table(
                cellText=subset.values,
                colLabels=subset.columns,
                cellLoc="center",
                loc="center",
            )

            # 生成文件名，如 output_0.png, output_1.png
            output_path = f"{directory}/output_{i // rows_per_image}.png"
            plt.savefig(output_path, bbox_inches="tight", dpi=300)

            # 关闭图像，释放内存
            plt.close(fig)

        print("所有图片已生成！")

class Download:
    def __init__(self):
        self.current_date = datetime.now() + timedelta(days=1)
        self.one_week_ago = self.current_date - timedelta(days=7)  # 计算一周之前的日期

    @staticmethod
    def download_with_retry(stock_code: str, max_retries=1, retry_delay=3):
        """
        带有重试机制的 akshare 股票数据下载
        不复权
        """
        current_date = datetime.now() + timedelta(days=1)
        end_date = current_date
        for attempt in range(max_retries):
            try:
                # 新浪(快)
                df = ak.stock_zh_a_daily(
                    symbol=stock_code,
                    start_date="2018-01-01",
                    end_date=end_date.strftime("%Y-%m-%d"),
                    # adjust="qfq" # qfq=前复权
                )

                # # 腾讯(慢)
                # df = ak.stock_zh_a_hist_tx(
                #     symbol=stock_code,
                #     start_date="2018-01-01",
                #     end_date=current_date.strftime("%Y-%m-%d"),
                #     adjust="",
                # )

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
        return None


class Final_process:
    @staticmethod
    def process_stock(stock_code:str, mode:str):
        '''不同模式下池化处理数据'''
        time.sleep(random.uniform(0.1, 0.2))  # 随机延时
        try:
            # 周死叉筛选
            if mode == "week_death":
                res = calculate.KDJ.get_week_death(stock_code)
                return res

            # 日死叉筛选
            elif mode == "daily_death":
                res = calculate.KDJ.get_day_death(stock_code)
                return res

            # 日金叉阶段筛选
            elif mode == "day_golden":
                res = calculate.KDJ.get_day_golden(stock_code)
                return res

            # J线掉头筛选
            elif mode == "J_turn_around":
                res = calculate.KDJ.get_day_J_turn_around(stock_code)
                return res

            # MACD: 1. 当日DIF刚上零线 OR 2. 当日MACD零线上金叉
            elif mode == "MACD_2_condition":
                res = calculate.Mix.get_MACD_2_condition(stock_code)
                return res

            # bol: 前次碰上轨，今日碰 中轨 or 下轨
            elif mode == "bol":
                res = calculate.Bol.get_bol(stock_code)
                return res

            # bol: 前次碰上轨，今日碰 下轨
            elif mode == "bol_lower":
                res = calculate.Bol.get_bol(stock_code, sub_mode="下")
                return res

            # 日MACD小于0的
            elif mode == "macd < 0":
                res = calculate.MACD.get_daily_macd_below0(stock_code)
                return res

            # 找出接近金叉的股票
            elif mode == "kdj_near_gold":
                res = calculate.KDJ.get_near_golden_cross(stock_code)
                return res

            # 找出J < K
            elif mode == "J_K":
                res = calculate.KDJ.get_J_K(stock_code)
                return res

            # 当日阴线 + 周kdj金叉小于三周
            elif mode == "阴线_周kdj":
                res = calculate.Mix.get_op1_1_cond3(stock_code)
                return res

            # # 一周内将分红股票 + 收益率降序排列
            # elif mode == "dividend":
            #     res = calculate.Dividend.get_dividend(stock_code)
            #     return res

            # 今日是第二根阴线 + 日MACD > 0 + '第三日'创新高 + 最近两天不能碰中轨
            elif mode == "op2_3":
                res = calculate.Mix.get_op2_cond3(stock_code)
                return res

            # 九转股
            elif mode =="nine":
                res = calculate.Nine.get_nine(stock_code)
                return res

            # 周KDJ 不死叉 or 周macd dif > 0 or 月kdj 不死叉 or 月macd dif > 0
            # +
            # 中轨向上 == 今日中轨大于昨日
            # +
            # 昨日最高价 > 今日最高价
            elif mode == "nine_several_condition":
                res = calculate.Nine.get_today(stock_code)
                return res

            # 利润 > 0 -> 保留
            # 利润 < 0 + 周&月macd dif > 0 -> 保留
            elif mode == "profit_check":
                res = calculate.Nine.check_net_profit_macd(stock_code)
                return res

            # 月macd dif刚刚大于0
            elif mode == "month_macd":
                res = calculate.MACD.check_monthly_macd_DIF(stock_code)
                return res

            # 月中线向上
            elif mode == "month_mid_line_upper":
                res = calculate.Bol.get_mid_up(stock_code)
                return res

            # 当月 月KDJ 金叉
            elif mode == "month_kdj_golden":
                res = calculate.KDJ.get_month_kdj_golden(stock_code)
                return res

            # 当日没有碰上轨
            elif mode == "not_touch_upper":
                res = calculate.Bol.get_not_touch_upper(stock_code)
                return res

            # pioneer1
            elif mode == "pioneer_1":
                res = calculate.Pioneer.pioneer_1(stock_code)
                return res

        except Exception as e:
            print(f"Error processing {stock_code}: {e}")
            return stock_code, []

        # 回收内存
        finally:
            gc.collect()

    @staticmethod
    def process_stocks(stock_list: list, mode: str):
        """线程池处理股票代码"""
        res_stock_list = []
        batch_size = 100  # 每批处理 100 支股票
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=10) as executor:  # 可以调整并发数量
                futures = {
                    executor.submit(Final_process.process_stock, stock, mode): stock
                    for stock in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        res_stock_list.append(result)
            gc.collect()  # 每批结束后强制回收
            print(
                f"Batch {i//batch_size + 1}/{len(stock_list)//batch_size + 1} completed, "
                f"memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB"
            )
        return res_stock_list


class Final_process_KDJ:
    @staticmethod
    def process_stock(stock_code: str):
        """处理单个股票，获取金叉和死叉日期"""
        time.sleep(random.uniform(0.1, 0.2))  # 随机延时
        try:
            golden_cross_dates = calculate.KDJ.get_recent_golden_cross_dates(stock_code)
            death_cross_dates = calculate.KDJ.get_recent_death_cross_dates(stock_code)
            return stock_code, golden_cross_dates, death_cross_dates
        except Exception as e:
            print(f"Error processing {stock_code}: {e}")
            return stock_code, [], []

    @staticmethod
    def process_stocks(stock_list: list):
        """多线程处理股票列表，获取所有股票的金叉和死叉日期"""
        all_cross_dates = {}  # 存储所有股票的交叉日期
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stock = {
                executor.submit(Final_process_KDJ.process_stock, stock): stock for stock in stock_list
            }

            for future in as_completed(future_to_stock):
                stock_code, golden_dates, death_dates = future.result()
                all_cross_dates[stock_code] = {
                    "Golden_Cross": golden_dates,
                    "Death_Cross": death_dates,
                }

        return all_cross_dates

class Download_KDJ:
    @staticmethod
    def download_with_retry(stock_code: str, max_retries=1, retry_delay=3):
        """
        带有重试机制的 akshare 股票数据下载
        不复权
        """
        current_date = datetime.now()
        end_date = current_date
        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_cdr_daily(
                    symbol=stock_code,
                    start_date="2018-01-01",
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

                else:
                    print(f"Max retries reached for {stock_code}. Skipping...")
        return pd.DataFrame()  # 如果失败，返回空 DataFrame

class Final_file_KDJ:
    @staticmethod
    def output_excel(all_cross_dates: dict):
        """
        导出金叉/死叉 Excel 文件。
        
        参数：
        - all_cross_dates: dict, 包含所有股票的交叉日期
        """
        filename = "./txt_lib/stock_name.txt"  # 股票名称文件
        stock_dict = Initialization.generate_stock_dict(filename)  # 读取股票代码-名称映射

        # 统计股票数量
        num_golden = 0
        num_death = 0
        golden_data = []
        death_data = []
        cross_type = ""

        for stock_code, cross_dates in all_cross_dates.items():

            golden_dates = cross_dates.get("Golden_Cross", [])
            death_dates = cross_dates.get("Death_Cross", [])

            if not golden_dates and not death_dates:
                continue

            cur_stock_code = str(stock_code[-6:])  # 取出六位股票代码
            stock_name = stock_dict.get(cur_stock_code, "未知")  # 获取股票名称

            # 只输出最近一周内的金叉或死叉
            if golden_dates:
                cross_type = "金叉"
                # 记录数据
                golden_data.append(
                    {
                        "股票代码": cur_stock_code,
                        "股票名称": stock_name,
                        f"一周内{cross_type}日期": golden_dates[
                            0
                        ],  # 记录最近一周的交叉日期
                    }
                )
                num_golden += 1  # 统计金叉

            if death_dates:
                cross_type = "死叉"
                # 记录数据
                death_data.append(
                    {
                        "股票代码": cur_stock_code,
                        "股票名称": stock_name,
                        f"一周内{cross_type}日期": death_dates[0],  # 记录最近一周的交叉日期
                    }
                )
                num_death += 1  # 统计死叉

        # 创建 DataFrame
        df = pd.DataFrame(golden_data)

        # 添加汇总信息
        golden_row = pd.DataFrame(
            {
                "股票代码": [f"总计金叉股票个数:{num_golden}"],
                "股票名称": [""],
                f"一周内{cross_type}日期": [""],
            }
        )

        df = pd.concat([golden_row, df], ignore_index=True)

        # 锁定输出地址
        output_file = "./output/golden_output/golden_cross_summary.xlsx"
        # 导出 Excel
        df.to_excel(output_file, index=False)

        # 创建 DataFrame
        df = pd.DataFrame(death_data)

        death_row = pd.DataFrame(
            {
                "股票代码": [f"总计死叉股票个数:{num_death}"],
                "股票名称": [""],
                f"一周内{cross_type}日期": [""],
            }
        )

        df = pd.concat([death_row, df], ignore_index=True)

        # 锁定输出地址
        output_file = "./output/death_output/death_cross_summary.xlsx"
        # 导出 Excel
        df.to_excel(output_file, index=False)

        print(f"Excel 文件已生成")

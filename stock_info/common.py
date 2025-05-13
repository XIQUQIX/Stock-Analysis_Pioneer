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

import calculate

class Initialization:
    @staticmethod
    # 生成股票代码 名称 词典
    def generate_stock_dict(filename: str):
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

class Final_file:
    @staticmethod
    # 输出最终excel文件
    def output_excel(stock_list: list, output_file: str, condition="金叉"):
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
            output_path = f"{directory}\output_{i // rows_per_image}.png"
            plt.savefig(output_path, bbox_inches="tight", dpi=300)

            # 关闭图像，释放内存
            plt.close(fig)

        print("所有图片已生成！")

class Download:
    def __init__(self):
        self.current_date = datetime.now() + timedelta(days=1)
        self.one_week_ago = self.current_date - timedelta(days=7)  # 计算一周之前的日期

    @staticmethod
    # 带有重试机制的 yfinance 数据下载函数。
    def download_with_retry(stock_code: str, max_retries=1, retry_delay=3):
        current_date = datetime.now() + timedelta(days=1)
        end_date = current_date
        for attempt in range(max_retries):
            try:
                # df = ak.stock_zh_a_cdr_daily(
                #     symbol=stock_code,
                #     start_date="2010-01-01",
                #     end_date=end_date.strftime("%Y-%m-%d"),
                # )
                df = ak.stock_zh_a_daily(
                    symbol=stock_code,
                    start_date="2010-01-01",
                    end_date=end_date.strftime("%Y-%m-%d"),
                    adjust="qfq",
                )  # qfq=前复权
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


class Final_process:
    @staticmethod
    def process_stock(stock_code:str, mode:str):
        time.sleep(random.uniform(0.1, 0.2))  # 随机延时
        try:
            # MACD金叉计算
            if mode == "macd":
                stock_code = calculate.MACD.get_macd(stock_code)
                return stock_code

            # 周死叉筛选
            elif mode == "week_death":
                stock_code = calculate.KDJ.get_week_death(stock_code)
                return stock_code

            # 日死叉筛选
            elif mode == "daily_death":
                stock_code = calculate.KDJ.get_day_death(stock_code)
                return stock_code

            # J线掉头筛选
            elif mode == "J_turn_around":
                stock_code = calculate.KDJ.get_day_J_turn_around(stock_code)
                return stock_code

            # MACD: 1. 当日DIF刚上零线 OR 2. 当日MACD零线上金叉
            elif mode == "MACD_2_condition":
                stock_code = calculate.Mix.get_MACD_2_condition(stock_code)
                return stock_code

        except Exception as e:
            print(f"Error processing {stock_code}: {e}")
            return stock_code, []

    @staticmethod
    # 线程池处理股票代码
    def process_stocks(stock_list: list, mode: str):
        all_golden_cross = []

        # 创建线程池，max_workers 可以根据需要调整
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(Final_process.process_stock, stock, mode): stock
                for stock in stock_list
            }
            for future in as_completed(futures):
                result = future.result()  # 这里获取 process_stock 的返回值
                if result:  # 确保返回值不是 None
                    all_golden_cross.append(result)

        return all_golden_cross

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
    # 带有重试机制的 akshare 数据下载函数。
    @staticmethod
    def download_with_retry(stock_code: str, max_retries=1, retry_delay=3):
        current_date = datetime.now() + timedelta(days=1)
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

class Final_file_KDJ:
    @staticmethod
    # 输出最终 Excel 文件
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

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

import common
import calculate

# 全局加载 pickle 文件
with open("./txt_lib/daily_df.pkl", "rb") as f:
    GLOBAL_STOCK_DATA = pickle.load(f)

class Dividend:
    @staticmethod
    def get_date_diff(stock_dividend: DataFrame):
        """获取股票距离分红最近日期"""
        # get今日 (date type)
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_format = "%Y-%m-%d"
        current_date_1 = datetime.strptime(current_date, date_format).date()

        # get股权登记日 (date type)
        register_day = stock_dividend["登记日"].iloc[-1]

        try:
            date_diff = register_day - current_date_1

        except Exception as e:
            return 0

        # 最近不分红
        if date_diff < timedelta(days=1):
            return 0

        return date_diff

    @staticmethod
    def get_dividend(stock_code: str):
        """获取最近分红的股票 及其 分红比例 和 收益率"""
        try:
            # 获取时间信息
            current_date = datetime.now().strftime("%Y-%m-%d")  # today
            end_date = current_date
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

            # 获取股票分红信息
            code_no_title = stock_code[2:]
            stock_detail_df = ak.stock_fhps_detail_em(symbol=code_no_title)
            stock_detail_df = stock_detail_df.rename(
                columns={
                    "现金分红-现金分红比例": "比例",
                    "现金分红-现金分红比例描述": "描述",
                    "股权登记日": "登记日",
                }
            )

            stock_dividend = stock_detail_df[["比例", "描述", "登记日"]]

            # 获取股票最近收盘价
            data = GLOBAL_STOCK_DATA
            df = pd.DataFrame(data[stock_code])

            date_diff = Dividend.get_date_diff(stock_dividend).days # 转换成int形式

            # 剔除最近不分红/大于30天股票
            if date_diff == 0 or date_diff > 30:
                return []

            ratio = stock_dividend["比例"].iloc[-1]  # 现金分红比例
            r_descr = stock_dividend["描述"].iloc[-1]
            register_day = stock_dividend["登记日"].iloc[-1]  # 股权登记日 (date type)

            # 收益率
            day_close = df["close"].iloc[-1]  # 取当日收盘为基础
            earn_rate = round(ratio / 10 / day_close, 3)
            res = [
                stock_code[2:],
                date_diff,
                ratio,
                r_descr,
                register_day,
                day_close,
                earn_rate,
            ]
            return res

        except Exception as e:
            return []

    @staticmethod
    def get_week_golden_cross(stock_code: str):
        """获取 周金叉 情况"""
        # 输入的str 是 '600029' 六位数字形式
        processed_code = common.Initialization.add_capital(stock_code)

        df = common.Download_KDJ.download_with_retry(processed_code)
        week_df = calculate.KDJ.calculate_weekly_kdj(df)
        week_cross = calculate.KDJ.find_kdj_golden_cross(week_df)

        return week_cross["Golden_Cross"].iloc[-1]  # 返回周金叉

    @staticmethod
    def get_target_df(cur_dividend: list[list]):
        """从分红股票中选出周金叉的 然后按照收益率降序排列"""
        columns = [
            "stock_code",
            "date_diff",
            "ratio",
            "ratio_descr",
            "register_day",
            "day_close",
            "earn_rate",
        ]
        df = pd.DataFrame(cur_dividend, columns=columns)
        df["week_golden"] = df["stock_code"].apply(Dividend.get_week_golden_cross)
        # 筛选 week_golden 为 True 的行
        final_df = df[df["week_golden"] == True]
        # 按earn_rate降序排列
        final_df = final_df.sort_values(by="earn_rate", ascending=False).reset_index(
            drop=True
        )
        return final_df

class Process:
    @staticmethod
    def process_stocks(stock_list: list, mode: str):
        """线程池处理股票代码"""
        res_stock_list = []
        batch_size = 100  # 每批处理 100 支股票
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=10) as executor:  # 可以调整并发数量
                futures = {
                    executor.submit(Process.process_stock, stock, mode): stock
                    for stock in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        res_stock_list.append(result)
            gc.collect()  # 每批结束后强制回收
            print(
                f"Batch {i//batch_size + 1} completed, memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB"
            )
        return res_stock_list

    @staticmethod
    def process_stock(stock_code: str, mode: str):
        """不同模式下池化处理数据"""
        time.sleep(random.uniform(0.1, 0.15))  # 随机延时

        if mode == "dividend":
            res =  Dividend.get_dividend(stock_code)
            return res

    @staticmethod
    def mk_pic(final_df:DataFrame):
        """每 50 行生成一张图片"""
        output_dir = "./output/dividend"

        # 每 50 行生成一张图片
        rows_per_image = 50
        num_rows = len(final_df)
        num_images = (num_rows + rows_per_image - 1) // rows_per_image  # 向上取整

        for i in range(num_images):
            # 获取当前 50 行的 stock_code 数据
            start_idx = i * rows_per_image
            end_idx = min(start_idx + rows_per_image, num_rows)
            stock_codes = final_df["stock_code"][start_idx:end_idx].tolist()

            # 创建图片
            fig, ax = plt.subplots(figsize=(4, len(stock_codes) * 0.3))  # 动态调整高度
            ax.axis("off")  # 关闭坐标轴

            # 创建表格数据（单列）
            table_data = [[code] for code in stock_codes]
            table = ax.table(
                cellText=table_data,
                colLabels=["stock_code"],  # 表头
                cellLoc="center",  # 单元格内容居中
                loc="center",  # 表格整体居中
                colWidths=[0.5],  # 调整列宽
                bbox=[0, 0, 1, 1]  # 表格充满整个图片
            )

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)  # 调整表格行高

            # 保存图片
            output_path = os.path.join(output_dir, f"output_{i}.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=100)
            plt.close()

        print(f"Generated {num_images} images in {output_dir}")

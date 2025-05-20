# %% [markdown]
# # 按需分类 KDJ日金叉 & KDJ日死叉 代码仓库

# %%
import calculate
import common
from pathlib import Path

# 给原始股票代码加上抬头；'600001' → 'sh600001'
def add_capital(input_file: str, output_file: str):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            stock_code = line.strip()
            if stock_code.startswith("6"):
                processed_code = f"sh{stock_code}"
            else:
                processed_code = f"sz{stock_code}"
            outfile.write(processed_code + "\n")

    print(f"Processed stock codes saved to {output_file}.")

# 复制一份昨日txt 留档
def copy_yest_txt(input_file: str, output_file: str):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            stock_code = line.strip()
            outfile.write(stock_code + "\n")
    print(f"Processed stock codes saved to {output_file}.")


dict_filename = "./txt_lib/stock_name.txt"
stock_dict = common.Initialization.generate_stock_dict(dict_filename)

input_file = "./txt_lib/KDJ日金.txt"
output_file = "./txt_lib/yest_KDJ日金.txt"
copy_yest_txt(input_file,output_file)
input_file = "./txt_lib/KDJ日死.txt"
output_file = "./txt_lib/yest_KDJ日死.txt"
copy_yest_txt(input_file, output_file)

# 清理两个文件夹
common.Final_file.clean_folder("./output/KDJ日金")
common.Final_file.clean_folder("./output/KDJ日死")
common.Final_file.clean_folder("./output/KDJ8")

# %% [markdown]
# ## 处理 KDJ日金 + 周死叉 → 弃

# %%
# 读取原始文件并处理
input_file = "./txt_lib/KDJ日金.txt"
output_file = "./txt_lib/processed_KDJ日金.txt"
add_capital(input_file, output_file)

file_name = output_file
with open(file_name, "r") as file:
    gold_stock_list = [line.strip() for line in file if line.strip()]

gold_to_del_stock_list = []
gold_to_del_stock_list = common.Final_process.process_stocks(gold_stock_list, mode="week_death")
# print("KDJ日金 因 周死叉 要删除的股票")
# print(f"总计股票数量:{len(gold_to_del_stock_list)}。")

# for stock_code in gold_to_del_stock_list:
#     cur_stock_code = stock_code[2:]
#     stock_name = stock_dict.get(cur_stock_code, "未知")  # 获取股票名称
#     print(f"{cur_stock_code}:{stock_name}")

# %% [markdown]
# ## 处理 KDJ日死 + 周死叉 → 弃

# %%
# 读取原始文件并处理
input_file = "./txt_lib/KDJ日死.txt"
output_file = "./txt_lib/processed_KDJ日死.txt"
add_capital(input_file, output_file)

file_name = output_file
with open(file_name, "r") as file:
    death_stock_list = [line.strip() for line in file if line.strip()]

death_to_del_stock_list = []
death_to_del_stock_list = common.Final_process.process_stocks(death_stock_list, mode="week_death")

# print("KDJ日死 因 周死叉 要删除的股票")
# print(f"总计股票数量:{len(death_to_del_stock_list)}。")

# for stock_code in death_to_del_stock_list:
#     cur_stock_code = stock_code[2:]
#     stock_name = stock_dict.get(cur_stock_code, "未知")  # 获取股票名称
#     print(f"{cur_stock_code}:{stock_name}")

# %% [markdown]
# ## 处理 KDJ日金 + 日死 → KDJ日死

# %%
gold_to_move_day_death_stock_list = []
gold_to_move_day_death_stock_list = common.Final_process.process_stocks(gold_stock_list, mode="daily_death")

# %% [markdown]
# ## 处理 KDJ日金 + Bol → 操作1.1

# %%
gold_to_1_1_list = []
gold_to_1_1_list = common.Final_process.process_stocks(gold_stock_list, mode="bol")

# %% [markdown]
# ## 处理 KDJ日金 + J线拐头 → KDJ8

# %%
gold_to_move_8_stock_list = []
gold_to_move_8_stock_list = common.Final_process.process_stocks(
    gold_stock_list, mode="J_turn_around"
)

# %% [markdown]
# ## KDJ日死 + Bol: 前次碰上轨，今日碰中轨 → 操作2

# %%
death_to_op2_list = []
death_to_op2_list = common.Final_process.process_stocks(death_stock_list, mode="bol")

# %% [markdown]
# # 图片 Excel生成

# %%
try:
    folder = Path("./output")
    folder.mkdir()
except Exception as e:
    pass

try:
    folder = Path("./output/KDJ日金")
    folder.mkdir()
    folder = Path("./output/KDJ日死")
    folder.mkdir()
    folder = Path("./output/KDJ8")
    folder.mkdir()
    folder = Path("./output/操作1.1")
    folder.mkdir()
    folder = Path("./output/操作2")
    folder.mkdir()
except Exception as e:
    pass

# %%
# 今日金叉数据
today_gold_txt_path = "./output/golden_output/today_gold_cross.txt"
today_gold_list = []
with open(today_gold_txt_path, "r") as file:
    for line in file:
        stock_code = line.strip()
        today_gold_list.append(stock_code)

# KDJ日金 文件生成
kdj_res = []
for stock_code in gold_stock_list:
    condition1 = stock_code not in gold_to_del_stock_list # 周死
    condition2 = stock_code not in gold_to_move_day_death_stock_list # 日死
    condition3 = stock_code not in gold_to_move_8_stock_list #拐头
    if condition1 and condition2 and condition3:
        kdj_res.append(stock_code)

# 加入今日金叉数据
for stock_code in today_gold_list:
    if stock_code not in kdj_res:
        kdj_res.append(stock_code)

output_file = './output/KDJ日金/KDJ日金.xlsx'
common.Final_file.output_excel(kdj_res,output_file,condition='日金')
common.Final_file.mk_pic(output_file)  # 生成所有图片

# %%
# 覆写KDJ日金txt
output_file = "./txt_lib/KDJ日金.txt"
with open(output_file, "w") as outfile1:
    for stock_code in kdj_res:
        processed_code = stock_code[2:]
        outfile1.write(processed_code + "\n")

print(f"Processed stock codes saved to {output_file}.")

# %%
# KDJ8 文件生成

# 读取昨日kdj8 txt
yest_kdj8_list = []
yest_kdj8_path = "./txt_lib/today_kdj8.txt"
with open(yest_kdj8_path, "r") as file:
    for line in file:
        stock_code = line.strip()
        if stock_code.startswith("6"):
            processed_code = f"sh{stock_code}"
        elif stock_code.startswith("0"):
            processed_code = f"sz{stock_code}"
        else:
            processed_code = stock_code
        yest_kdj8_list.append(processed_code)

# 筛出日死代码
kdj8_to_move_day_death_stock_list = []
kdj8_to_move_day_death_stock_list = common.Final_process.process_stocks(
    yest_kdj8_list, mode="daily_death"
)

# 筛出周死代码
kdj8_to_del_stock_list = []
kdj8_to_del_stock_list = common.Final_process.process_stocks(
    yest_kdj8_list, mode="week_death"
)

today_kdj8 = []

for stock_code in yest_kdj8_list:
    condition1 = stock_code not in kdj8_to_del_stock_list # 周死
    condition2 = stock_code not in kdj8_to_move_day_death_stock_list # 日死
    condition3 = stock_code not in gold_to_move_8_stock_list # 今日
    if condition1 and condition2 and condition3:
        today_kdj8.append(stock_code)

today_kdj8 += gold_to_move_8_stock_list

# 最终输出
output_file = "./output/KDJ8/KDJ8.xlsx"
common.Final_file.output_excel(today_kdj8, output_file, condition="8")
common.Final_file.mk_pic(output_file)  # 生成所有图片

# 生成今日 KDJ8 TXT
today_kdj8_txt_path = "./txt_lib/today_kdj8.txt"
with open(today_kdj8_txt_path, "w") as file:
    for item in today_kdj8:
        file.write(item + "\n")  # 每个元素后加换行符

print("今日KDJ8 txt文件已生成。")

# %%
# KDJ日死 文件生成
kdj_death = []

# 今日死叉数据
today_death_txt_path = "./output/death_output/today_death_cross.txt"
today_death_list = []
with open(today_death_txt_path, "r") as file:
    for line in file:
        stock_code = line.strip()
        today_death_list.append(stock_code)

for stock_code in death_stock_list:
    condition1 = stock_code not in death_to_del_stock_list
    condition2 = stock_code not in death_to_op2_list
    if condition1 and condition2:
        kdj_death.append(stock_code)

# 加入今日死叉数据
for stock_code in today_death_list:
    if stock_code not in kdj_death:
        kdj_death.append(stock_code)
        
# 加入今日kdj8日死数据
for stock_code in kdj8_to_move_day_death_stock_list:
    if stock_code not in kdj_death:
        kdj_death.append(stock_code)

output_file = "./output/KDJ日死/KDJ日死.xlsx"
common.Final_file.output_excel(kdj_death, output_file, condition="日死")
common.Final_file.mk_pic(output_file)  # 生成所有图片

# %%
output_file = "./output/操作2/操作2.xlsx"
common.Final_file.output_excel(death_to_op2_list, output_file, condition="bol")
common.Final_file.mk_pic(output_file)  # 生成所有图片

# %%
# 覆写KDJ日死txt
output_file = "./txt_lib/KDJ日死.txt"
with open(output_file, "w") as outfile1:
    for stock_code in kdj_death:
        processed_code = stock_code[2:]
        outfile1.write(processed_code + "\n")

print(f"Processed stock codes saved to {output_file}.")


# %% [markdown]
# ## TEST

# %%
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd
import numpy as np

import calculate

# 获取数据
stock_code = "sh600519"
current_date = datetime.now()
end_date = current_date

df = ak.stock_zh_a_daily(
    symbol=stock_code,
    start_date="2010-01-01",
    end_date=end_date.strftime("%Y-%m-%d"),
    adjust="qfq",
)

# 预处理数据
df.index = pd.to_datetime(df["date"])
df = df.sort_index()

df = calculate.Bol.calculate_bollinger_bands(df)
touch_records = calculate.Bol.detect_bollinger_touch(df)  # 获取碰轨事件

# %%
current_date = datetime.now().strftime("%Y-%m-%d")  # today

op2 = [] # 移至操作2

last_touch_date = touch_records[-1][0]
if last_touch_date == current_date:
    last_touch_type = touch_records[-1][1] #本回轨道
    bef_last_touch_type = touch_records[-2][1] #上回轨道
    if last_touch_type == "中轨" and bef_last_touch_type == "上轨":
        op2.append(stock_code)



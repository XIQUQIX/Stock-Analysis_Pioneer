# 沪深股市
# Daily Stock Analysis with Golden Cross Detection of KDJ and MACD

## Overview
This project is designed to analyze stock data, identify golden cross patterns, MACD data, K line type, etc.

Further checks are made to filter stocks in KDJ daily golden cross and KDJ daily death to operation 1.1 and operation 2 for final manual check.

The analysis is outputed in `output` folder. Ecah folder includes:

An Excel file:
- Stock code
- Stock name
- Dates of golden crosses in the past week
- Total number of stocks with golden crosses

And the related png files:
- each picture is grouped by at most 50 stock codes
- for user to load into their stock softwares

The project is implemented in Python and utilizes libraries including `akshare`, `pandas`, `matplot` and `openpyxl`.
Lib `yfinance` is no longer used due to its unstability.

## Features
1. **Golden Cross Detection**: Identify daily and weekly golden cross patterns in stock data.
2. **MACD Monthly Analysis**: Filter stocks that experienced a MACD monthly golden cross in current month and its DIF is above 0.
3. **Excel Report Generation**: Output the results to an Excel file with a summary of the total number of stocks meeting the criteria.
4. **Error Handling for Data Retrieval**: Automatically retries data download if an error occurs.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required libraries:
  - `matplot`
  - `akshare`
  - `pandas`
  - `numpy`

You can install the required libraries using:
```bash
pip install matplot
pip install akshare
pip install pandas
```

## Usage

### 1. Data Preparation
- **Input Files**:
  - Every required file is stored in `txt_lib` folder.
    - Yet, the stock name is not updated and some may even have been delisted. Please check out and you could update that by editing the stock_name.txt. 

### 2. Running the Analysis
The analysis script can be scheduled to run daily at 8:00 AM (e.g., using Windows Task Scheduler). The main script to execute is `kdj_整合.ipynb` or its converted Python script.

#### Steps:
1. Convert the Jupyter Notebook to a Python script if necessary:
   ```bash
   jupyter nbconvert --to script kdj_整合.ipynb
   jupyter nbconvert --to script kdj_分类筛选.ipynb
   ```

   Or, you could use the export function in Vscode to get a py file from `kdj_整合.ipynb`.
   
3. Execute the script.

### 3. Output
- The script generates an `output` folder:
  - KDJ日金
  - KDJ日死
  - 操作1.1
  - 操作2
  - KDJ8
  - death_output
  - golden_output
  - month_macd

## 输出文件规则一览

### 1 Golden Output
- 日金叉

- 周金叉

- 今日股价上涨

### 2 Death Output
- 日死叉

- 周金叉

- 今日股价上涨

### 3 KDJ日金
- 今日金叉

### 4 KDJ日死
- 今日死叉

### 5 操作1.1
- ***今日金叉***
    - MACD

        DIF刚刚>0

            DIF 今日大于0， 昨日小于0

        零线上金叉

        今日不能碰上轨

- ***KDJ日金(今日数据)***
    - 布林线

        上次碰轨事件——碰上轨

        今日碰中轨/下轨

        昨日最高价 > 今日最高价
    
    - 阴线

        今日是阴线 + 昨日是阳线

        周kdj金叉三周内

        昨日阳线碰上轨

        今日最高价 小于 昨日最高价

### 6 KDJ8
- ***KDJ日金(昨日数据)***
    - J线拐头

        今日J值小于昨日J值
        
        今日J值大于今日K值

### 7 操作2
- ***KDJ日死(昨日数据)***
    - 布林线
    
        上次碰轨事件——碰上轨
        
        今日碰下轨

    - 接近金叉

        昨日D线K线差值 大于 今日D线K线差值

        DIF > 0
    
        MACD > 0

- ***KDJ日死(今日数据)***

    - 阴线

        今日是第二根阴线

        日MACD > 0

        '第三日'创最近十日新高

        最近两天不能碰中轨


### 8 弃用
- ***KDJ日金(昨日数据)***

    - 周死叉

    - macd < 0

    - 当日处于死叉阶段

- ***KDJ日死(昨日数据)***

    - 周死叉

    - 当日处于金叉阶段

### 9 dividend

    - 最近30天内会分红

    - 周KDJ金叉

## Some Core Functions
### Function in `calculate.py`
- KDJ class
  - `calculate_kdj(df: DataFrame, n=9)`
    - Calculates the daily KDJ.
    - window = n = 9 is set according to the normal calculation

  - `find_kdj_golden_cross`
    - Calculate golden cross, death cross and  rise/ fall.

  - `get_recent_golden_cross_dates(stock_code)`
    - Retrieves golden cross dates of a given stock and the date is within this week.

- MACD class
  - `calculate_macd(df: DataFrame, short=12, long=26, signal=9)`
    - Calculates the daily MACD.
    - short=12, long=26, signal=9 are all set according to the normal calculation

- Bol class
  - `calculate_bollinger_bands`
    -   calculate the exacat upper, mid and lower Bollinger band
    -   the lower band is ￥0.03 upper due to special needs 


- Mix class
  - functions within this class are mixture usage of upper classes, which are highly customized
  - `get_k_line_type`
    - determine today's K line type (negative/ positive)

### Function in common.py

## Error Handling
- **Data Retrieval**: Implements retry logic for downloading stock data using `akshare` to handle occasional `JSONDecodeError` or other network issues.
- **Fallback Values**: Ensures the analysis continues even if some stocks have incomplete data.

## Scheduling with Windows Task Scheduler
To schedule the script to run every weekday at 8:00 AM:
1. Open Windows Task Scheduler.
2. Create a new task:
   - Set the trigger to daily at 4:00 PM.
   - Set the action to run the Python script (`kdj_金叉+死叉_ak.py`).
   - Ensure the correct Python interpreter is used.
3. Save and enable the task.

## License
This project is licensed under the MIT License. See `LICENSE` for details.



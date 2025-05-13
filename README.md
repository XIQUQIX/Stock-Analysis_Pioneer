# 沪深股市
# Daily Stock Analysis with Golden Cross Detection of KDJ and MACD

## Overview
This project is designed to analyze stock data, identify golden cross patterns, and filter stocks that experienced a MACD month golden cross and the DIF is above 0. The analysis outputs an Excel file containing:
- Stock code
- Stock name
- Dates of golden crosses in the past week
- Total number of stocks with golden crosses

And the related png file for user to load into their stock softwares
- each picture is grouped by at most 50 stock codes

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
  - `stock_codes.txt`: A text file containing stock codes, one per line.
  - `stock_name.txt`: A text file mapping stock codes to stock names.

### 2. Running the Analysis
The analysis script can be scheduled to run daily at 8:00 AM (e.g., using Windows Task Scheduler). The main script to execute is `test.ipynb` or its converted Python script.

#### Steps:
1. Convert the Jupyter Notebook to a Python script if necessary:
   ```bash
   jupyter nbconvert --to script kdj_金叉+死叉_ak.ipynb
   jupyter nbconvert --to script kdj_分类筛选.ipynb
   ```
2. Execute the script:

### 3. Output
- The script generates an Excel file named `golden_cross_report.xlsx` in output folder, containing the following columns:
  - **Stock Code**: The six-digit stock code.
  - **Stock Name**: The corresponding name of the stock.
  - **Golden Cross Dates**: Dates within the past week when a golden cross occurred.
  - **Total Count**: The total number of stocks meeting the criteria, displayed at the top of the sheet.

## Key Functions
### `get_recent_golden_cross_dates(stock_code)`
Retrieves golden cross dates for a given stock.
- Downloads historical stock data using `akshare`.
- Calculates KDJ and MACD indicators.
- Filters golden cross dates within the last week.

### `generate_excel_report(data, filename)`
Generates the Excel report summarizing the analysis.

## Error Handling
- **Data Retrieval**: Implements retry logic for downloading stock data using `yfinance` to handle occasional `JSONDecodeError` or other network issues.
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



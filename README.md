# 缅A golden_cross for KDJ and MACD
# Daily Stock Analysis with Golden Cross Detection

## Overview
This project is designed to analyze stock data, identify golden cross patterns, and filter stocks that experienced a MACD daily golden cross within the past week. The analysis outputs an Excel file containing:
- Stock code
- Stock name
- Dates of golden crosses in the past week
- Total number of stocks with golden crosses

The project is implemented in Python and utilizes libraries like `yfinance`, `pandas`, and `openpyxl`.

## Features
1. **Golden Cross Detection**: Identify weekly golden cross patterns in stock data.
2. **MACD Daily Analysis**: Filter stocks that experienced a MACD daily golden cross in the past week.
3. **Excel Report Generation**: Output the results to an Excel file with a summary of the total number of stocks meeting the criteria.
4. **Error Handling for Data Retrieval**: Automatically retries data download if an error occurs.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required libraries:
  - `yfinance`
  - `pandas`
  - `openpyxl`
  - `numpy`

You can install the required libraries using:
```bash
pip install yfinance
```

## Usage

### 1. Data Preparation
- **Input Files**:
  - `stock_code_1.txt`: A text file containing stock codes, one per line.
  - `stock_name.txt`: A text file mapping stock codes to stock names.

### 2. Running the Analysis
The analysis script can be scheduled to run daily at 8:00 AM (e.g., using Windows Task Scheduler). The main script to execute is `test.ipynb` or its converted Python script.

#### Steps:
1. Convert the Jupyter Notebook to a Python script if necessary:
   ```bash
   jupyter nbconvert --to script test.ipynb
   ```
2. Execute the script:
   ```bash
   python test.py
   ```

### 3. Output
- The script generates an Excel file named `golden_cross_report.xlsx`, containing the following columns:
  - **Stock Code**: The six-digit stock code.
  - **Stock Name**: The corresponding name of the stock.
  - **Golden Cross Dates**: Dates within the past week when a golden cross occurred.
  - **Total Count**: The total number of stocks meeting the criteria, displayed at the top of the sheet.

## Key Functions
### `get_recent_golden_cross_dates(stock_code)`
Retrieves golden cross dates for a given stock.
- Downloads historical stock data using `yfinance`.
- Calculates KDJ and MACD indicators.
- Filters golden cross dates within the last week.

### `filter_macd_golden_cross_in_last_week(data, all_golden_cross_dates)`
Filters stocks from `all_golden_cross_dates` that experienced a MACD daily golden cross in the last week.

### `generate_excel_report(data, filename)`
Generates the Excel report summarizing the analysis.

## Error Handling
- **Data Retrieval**: Implements retry logic for downloading stock data using `yfinance` to handle occasional `JSONDecodeError` or other network issues.
- **Fallback Values**: Ensures the analysis continues even if some stocks have incomplete data.

## Scheduling with Windows Task Scheduler
To schedule the script to run every weekday at 8:00 AM:
1. Open Windows Task Scheduler.
2. Create a new task:
   - Set the trigger to daily at 8:00 AM.
   - Set the action to run the Python script (`test.py`).
   - Ensure the correct Python interpreter is used.
3. Save and enable the task.

## Troubleshooting
- **Data Download Issues**: If a stock fails to download, the script retries automatically.
- **Empty Results**: Ensure the stock codes in `stock_code_1.txt` are valid and active.

## License
This project is licensed under the MIT License. See `LICENSE` for details.



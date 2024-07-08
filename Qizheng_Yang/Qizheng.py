import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Login to baostock
bs.login()

# Function to fetch historical data
def get_data(code, start_date, end_date):
    rs = bs.query_history_k_data_plus(code,
                                      "date,code,close",
                                      start_date=start_date, end_date=end_date,
                                      frequency="d", adjustflag="3")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    return df

# Define stock codes
stock_codes = [
    'sh.600000', 'sh.600009', 'sh.600010', 'sh.600011', 'sh.600015',
    'sh.600016', 'sh.600018', 'sh.600019', 'sh.600023', 'sh.600025',
    'sh.600026', 'sh.600027', 'sh.600028', 'sh.600029', 'sh.600030',
    'sh.600031', 'sh.600036', 'sh.600039', 'sh.600048', 'sh.600050'
]

# Fetch data for all stocks
stock_data = {code: get_data(code, "2014-01-01", "2024-06-30") for code in stock_codes}

# Prepare a DataFrame to store results
results = []

# Initialize a DataFrame to store cumulative PnL
cumulative_pnl = pd.DataFrame()

# Perform pair trading analysis for each pair
for i in range(len(stock_codes)):
    for j in range(i + 1, len(stock_codes)):
        stock_A_code = stock_codes[i]
        stock_B_code = stock_codes[j]

        stock_A = stock_data[stock_A_code]
        stock_B = stock_data[stock_B_code]

        # Merge the data on the date
        merged_data = pd.merge(stock_A[['date', 'close']], stock_B[['date', 'close']], on='date', suffixes=('_A', '_B'))

        # Perform cointegration test
        coint_result = coint(merged_data['close_A'], merged_data['close_B'])
        p_value = coint_result[1]

        # Calculate the spread and hedge ratio if cointegrated
        if p_value < 0.05:
            hedge_ratio = sm.OLS(merged_data['close_A'], merged_data['close_B']).fit().params[0]
            merged_data['spread'] = merged_data['close_A'] - hedge_ratio * merged_data['close_B']

            # Signal generation
            mean_spread = merged_data['spread'].mean()
            std_spread = merged_data['spread'].std()
            merged_data['z_score'] = (merged_data['spread'] - mean_spread) / std_spread

            # Define entry and exit signals
            entry_threshold = 1.5
            exit_threshold = 0

            # Long entry: Z-score < -entry_threshold
            merged_data['long_entry'] = merged_data['z_score'] < -entry_threshold
            # Short entry: Z-score > entry_threshold
            merged_data['short_entry'] = merged_data['z_score'] > entry_threshold
            # Exit: Z-score reverts to zero
            merged_data['exit'] = abs(merged_data['z_score']) < exit_threshold

            # Initialize position and PnL columns
            merged_data['position'] = 0
            merged_data['pnl'] = 0

            # Calculate positions and PnL
            position = 0
            for k in range(1, len(merged_data)):
                if merged_data.loc[k, 'long_entry']:
                    position = 1  # Go long
                elif merged_data.loc[k, 'short_entry']:
                    position = -1  # Go short
                elif merged_data.loc[k, 'exit']:
                    position = 0  # Exit position

                merged_data.loc[k, 'position'] = position
                spread_change = merged_data.loc[k, 'spread'] - merged_data.loc[k - 1, 'spread']
                transaction_cost = 0.004 * abs(position - merged_data.loc[k - 1, 'position']) * merged_data.loc[k - 1, 'spread']
                merged_data.loc[k, 'pnl'] = position * spread_change - transaction_cost

            # Calculate cumulative PnL
            merged_data['cumulative_pnl'] = merged_data['pnl'].cumsum()

            # Sum cumulative PnL across all pairs
            if cumulative_pnl.empty:
                cumulative_pnl = merged_data[['date', 'cumulative_pnl']].copy()
                cumulative_pnl.rename(columns={'cumulative_pnl': f'cumulative_pnl_{stock_A_code}_{stock_B_code}'}, inplace=True)
            else:
                cumulative_pnl = pd.merge(cumulative_pnl, merged_data[['date', 'cumulative_pnl']], on='date', how='outer')
                cumulative_pnl.rename(columns={'cumulative_pnl': f'cumulative_pnl_{stock_A_code}_{stock_B_code}'}, inplace=True)

            # Save results
            results.append({
                'Stock_A': stock_A_code,
                'Stock_B': stock_B_code,
                'P-Value': p_value,
                'Hedge_Ratio': hedge_ratio,
                'Mean_Spread': mean_spread,
                'Std_Spread': std_spread,
                'Cumulative_PnL': merged_data['cumulative_pnl'].iloc[-1],
                'Data': merged_data
            })

# Create a single PnL curve by summing all individual cumulative PnLs
cumulative_pnl['total_cumulative_pnl'] = cumulative_pnl.iloc[:, 1:].sum(axis=1)

# Calculate performance summary statistics
total_return = cumulative_pnl['total_cumulative_pnl'].iloc[-1]
num_years = (cumulative_pnl['date'].iloc[-1] - cumulative_pnl['date'].iloc[0]).days / 365.25
annualized_return = total_return / num_years
daily_returns = cumulative_pnl['total_cumulative_pnl'].diff().dropna()
annualized_volatility = daily_returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

# Calculate maximum drawdown
cumulative_pnl['max_pnl'] = cumulative_pnl['total_cumulative_pnl'].cummax()
cumulative_pnl['drawdown'] = cumulative_pnl['total_cumulative_pnl'] - cumulative_pnl['max_pnl']
max_drawdown = cumulative_pnl['drawdown'].min()

# Print performance summary statistics
print(f"Total Return: {total_return}")
print(f"Annualized Return: {annualized_return}")
print(f"Annualized Volatility: {annualized_volatility}")
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Maximum Drawdown: {max_drawdown}")

# Plot the single PnL curve for the out of sample data (2022-01-02 to 2024-06-30)
cumulative_pnl_out_of_sample = cumulative_pnl[(cumulative_pnl['date'] >= '2022-01-02') & (cumulative_pnl['date'] <= '2024-06-30')]

plt.figure(figsize=(15, 7))
plt.plot(cumulative_pnl_out_of_sample['date'], cumulative_pnl_out_of_sample['total_cumulative_pnl'], label='Total Cumulative PnL')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.title('Total Cumulative PnL for All Pairs (Out of Sample Data)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_cumulative_pnl_out_of_sample.png')
plt.show()

# Logout from baostock
bs.logout()
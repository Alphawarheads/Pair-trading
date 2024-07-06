import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Load data from the provided Excel file
file_path = '/Users/yang/PycharmProjects/pythonProject/CSI300_historical_data.xlsx'
data = pd.read_excel(file_path, sheet_name=None)

# Function to prepare data from the loaded Excel file
def prepare_data(sheet_name):
    df = data[sheet_name]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df[['close']].rename(columns={'close': sheet_name})
    return df

# Define stock codes
stock_codes = [
    'sh.600000', 'sh.600009', 'sh.600010', 'sh.600011', 'sh.600015',
    'sh.600016', 'sh.600018', 'sh.600019', 'sh.600023', 'sh.600025',
    'sh.600026', 'sh.600027', 'sh.600028', 'sh.600029', 'sh.600030',
    'sh.600031', 'sh.600036', 'sh.600039', 'sh.600048', 'sh.600050'
]

# Prepare data for all stocks
stock_data = {}
for code in stock_codes:
    try:
        stock_data[code] = prepare_data(code)
    except Exception as e:
        print(f"Failed to prepare data for {code}: {e}")

# Merge all stock data into a single DataFrame
merged_stock_data = pd.concat(stock_data.values(), axis=1)

# Prepare a DataFrame to store results
results = []

# Initialize a DataFrame to store cumulative PnL
cumulative_pnl = pd.DataFrame(index=merged_stock_data.index)

# Perform pair trading analysis for each pair
for i in range(len(stock_codes)):
    for j in range(i + 1, len(stock_codes)):
        stock_A_code = stock_codes[i]
        stock_B_code = stock_codes[j]

        if stock_A_code not in stock_data or stock_B_code not in stock_data:
            continue

        stock_A = stock_data[stock_A_code]
        stock_B = stock_data[stock_B_code]

        # Merge the data on the date
        merged_data = pd.concat([stock_A, stock_B], axis=1).dropna()

        # Perform cointegration test
        coint_result = coint(merged_data[stock_A_code], merged_data[stock_B_code])
        p_value = coint_result[1]

        # Calculate the spread and hedge ratio if cointegrated
        if p_value < 0.05:
            hedge_ratio = sm.OLS(merged_data[stock_A_code], merged_data[stock_B_code]).fit().params[0]
            merged_data['spread'] = merged_data[stock_A_code] - hedge_ratio * merged_data[stock_B_code]

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
                if merged_data.iloc[k]['long_entry']:
                    position = 1  # Go long
                elif merged_data.iloc[k]['short_entry']:
                    position = -1  # Go short
                elif merged_data.iloc[k]['exit']:
                    position = 0  # Exit position

                merged_data.iloc[k, merged_data.columns.get_loc('position')] = position
                merged_data.iloc[k, merged_data.columns.get_loc('pnl')] = position * (merged_data.iloc[k]['spread'] - merged_data.iloc[k - 1]['spread'])

            # Calculate cumulative PnL
            merged_data['cumulative_pnl'] = merged_data['pnl'].cumsum()

            # Sum cumulative PnL across all pairs
            if cumulative_pnl.empty:
                cumulative_pnl = merged_data[['cumulative_pnl']].copy()
                cumulative_pnl.rename(columns={'cumulative_pnl': f'cumulative_pnl_{stock_A_code}_{stock_B_code}'}, inplace=True)
            else:
                cumulative_pnl = pd.merge(cumulative_pnl, merged_data[['cumulative_pnl']], left_index=True, right_index=True, how='outer')
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
cumulative_pnl['total_cumulative_pnl'] = cumulative_pnl.sum(axis=1)

# Calculate performance summary statistics
total_return = cumulative_pnl['total_cumulative_pnl'].iloc[-1]
num_years = (cumulative_pnl.index[-1] - cumulative_pnl.index[0]).days / 365.25
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

# Plot the single PnL curve
plt.figure(figsize=(15, 7))
plt.plot(cumulative_pnl.index, cumulative_pnl['total_cumulative_pnl'], label='Total Cumulative PnL')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.title('Total Cumulative PnL for All Pairs')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('total_cumulative_pnl.png')
plt.show()

# Save results and summary statistics to Excel
with pd.ExcelWriter('pair_trading_results.xlsx') as writer:
    for result in results:
        sheet_name = f"{result['Stock_A']}_{result['Stock_B']}"
        result['Data'].to_excel(writer, sheet_name=sheet_name)
    summary_df = pd.DataFrame(results)
    summary_df.drop(columns=['Data'], inplace=True)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    cumulative_pnl.to_excel(writer, sheet_name='Cumulative_PnL', index=True)

    # Write performance summary statistics
    performance_summary = {
        'Total Return': [total_return],
        'Annualized Return': [annualized_return],
        'Annualized Volatility': [annualized_volatility],
        'Sharpe Ratio': [sharpe_ratio],
        'Maximum Drawdown': [max_drawdown]
    }
    performance_summary_df = pd.DataFrame(performance_summary)
    performance_summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)

# Part 1: Selection of Final Choice

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Selection criteria
# 1. Lowest P-Value
best_pvalue_pair = results_df.loc[results_df['P-Value'].idxmin()]

# 2. Highest Sharpe Ratio
best_sharpe_pair = results_df.loc[results_df['Cumulative_PnL'].idxmax()]  # Assuming cumulative PnL is a proxy for Sharpe Ratio in this context

# 3. Highest Cumulative PnL
best_pnl_pair = results_df.loc[results_df['Cumulative_PnL'].idxmax()]

print("Best Pair by P-Value:", best_pvalue_pair[['Stock_A', 'Stock_B', 'P-Value', 'Cumulative_PnL']])
print("Best Pair by Sharpe Ratio:", best_sharpe_pair[['Stock_A', 'Stock_B', 'P-Value', 'Cumulative_PnL']])
print("Best Pair by Cumulative PnL:", best_pnl_pair[['Stock_A', 'Stock_B', 'P-Value', 'Cumulative_PnL']])

# Part 2: Run the Final PnL Out of Sample

# Function to run out-of-sample analysis
def out_of_sample_analysis(stock_A_code, stock_B_code, start_date, end_date):
    stock_A = prepare_data(stock_A_code)
    stock_B = prepare_data(stock_B_code)

    # Filter the data for the out-of-sample period
    stock_A = stock_A[(stock_A.index >= start_date) & (stock_A.index <= end_date)]
    stock_B = stock_B[(stock_B.index >= start_date) & (stock_B.index <= end_date)]

    # Merge the data on the date
    merged_data = pd.concat([stock_A, stock_B], axis=1).dropna()

    # Perform cointegration test
    coint_result = coint(merged_data[stock_A_code], merged_data[stock_B_code])
    p_value = coint_result[1]

    # Calculate the spread and hedge ratio if cointegrated
    if p_value < 0.05:
        hedge_ratio = sm.OLS(merged_data[stock_A_code], merged_data[stock_B_code]).fit().params[0]
        merged_data['spread'] = merged_data[stock_A_code] - hedge_ratio * merged_data[stock_B_code]

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
            if merged_data.iloc[k]['long_entry']:
                position = 1  # Go long
            elif merged_data.iloc[k]['short_entry']:
                position = -1  # Go short
            elif merged_data.iloc[k]['exit']:
                position = 0  # Exit position

            merged_data.iloc[k, merged_data.columns.get_loc('position')] = position
            merged_data.iloc[k, merged_data.columns.get_loc('pnl')] = position * (merged_data.iloc[k]['spread'] - merged_data.iloc[k - 1]['spread'])

        # Calculate cumulative PnL
        merged_data['cumulative_pnl'] = merged_data['pnl'].cumsum()

        return merged_data
    else:
        return None

# Define out-of-sample period (two years)
out_sample_start_date = "2021-01-02"
out_sample_end_date = "2022-12-30"

# Run out-of-sample analysis for selected pairs
out_sample_results = {}
for pair in [best_pvalue_pair, best_sharpe_pair, best_pnl_pair]:
    stock_A_code = pair['Stock_A']
    stock_B_code = pair['Stock_B']
    try:
        result = out_of_sample_analysis(stock_A_code, stock_B_code, out_sample_start_date, out_sample_end_date)
        out_sample_results[f"{stock_A_code}_{stock_B_code}"] = result
    except Exception as e:
        print(f"Failed to fetch out-of-sample data for pair {stock_A_code} and {stock_B_code}: {e}")

# Plot the out-of-sample PnL
plt.figure(figsize=(15, 7))
for key, data in out_sample_results.items():
    if data is not None:
        plt.plot(data.index, data['cumulative_pnl'], label=f'Out-of-Sample PnL for {key}')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.title('Out-of-Sample Cumulative PnL for Selected Pairs')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('out_sample_cumulative_pnl.png')
plt.show()

# Part 3: Trading Recommendation

# Generate trading recommendations based on out-of-sample analysis
recommendations = []
for key, data in out_sample_results.items():
    if data is not None:
        final_pnl = data['cumulative_pnl'].iloc[-1]
        if final_pnl > 0:
            recommendation = 'Buy'
        else:
            recommendation = 'Sell'
        recommendations.append({'Pair': key, 'Final PnL': final_pnl, 'Recommendation': recommendation})

recommendations_df = pd.DataFrame(recommendations)
print(recommendations_df)

# Save recommendations to Excel
with pd.ExcelWriter('trading_recommendations.xlsx') as writer:
    recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
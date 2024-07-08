import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 设置参数
file_path = 'CSI300index.xlsx'
sheet_name = 'Sheet1'
start_date_train = pd.to_datetime('2013-01-01')
end_date_train = pd.to_datetime('2021-12-31')
start_date_test = pd.to_datetime('2013-01-01')
end_date_test = pd.to_datetime('2019-12-31')
initial_capital = 10000

df = pd.read_excel(file_path, sheet_name=sheet_name, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x), engine='openpyxl')

df.columns = df.columns.str.strip()
code_list = df['Code'].unique()
result = pd.DataFrame()

# 遍历所有的股票代码
for code in tqdm(code_list, desc="Processing stocks"):
    try:
        # 选择特定股票代码的数据
        stock_df = df[df['Code'] == code]
        stock_df.loc[:, 'Stock'] = code
        
        # 选择 'Date', 'Close', 和 'Stock' 列
        stock_df = stock_df[['Date', 'Close', 'Stock']]
        
        # 填补上市之前的缺失值
        stock_df = stock_df.set_index('Date')
        stock_df = stock_df.reindex(pd.date_range(start=start_date_train, end=end_date_test, freq='B'))
        stock_df['Stock'] = code
        
        # 使用前向填充和后向填充填补缺失值
        stock_df['Close'] = stock_df['Close'].ffill().bfill()
        
        # 将结果追加到result数据框
        result = pd.concat([result, stock_df.reset_index()])
    except KeyError:
        print(f"Error processing {code}")
        continue

# 重置索引并重命名列
result.reset_index(drop=True, inplace=True)
result.rename(columns={'index': 'Date'}, inplace=True)

# 获取历史价格数据
data = result.pivot(index='Date', columns='Stock', values='Close')

# 过滤缺失值过多的股票
min_non_missing_values = len(pd.date_range(start=start_date_train, end=end_date_test, freq='B')) * 0.75
data = data.dropna(thresh=min_non_missing_values, axis=1)

# 限制变量数量在12个以内
data = data.iloc[:, :12]

# 分割训练数据和测试数据
train_data = data[(data.index >= start_date_train) & (data.index <= end_date_train)]
test_data = data[(data.index >= start_date_test) & (data.index <= end_date_test)]

# 检查平稳性
def check_stationarity(data, significance_level=0.05):
    stationary_codes = []
    for col in tqdm(data.columns, desc="Checking stationarity"):
        if data[col].apply(lambda x: np.isinf(x) or np.isnan(x)).any():
            print(f"Skipping {col} due to inf or nan values")
            continue
        try:
            _, pvalue, _, _, _, _ = sm.tsa.adfuller(data[col])
            if pvalue < significance_level:
                stationary_codes.append(col)
        except ValueError:
            print(f"Error processing {col}")
            continue
    return stationary_codes

# 找出协整配对
def find_cointegrated_pairs(data, significance_level=0.05):
    stationary_codes = check_stationarity(data, significance_level)
    data = data[stationary_codes]
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []

    for i in tqdm(range(n), desc="Checking pairs"):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            score, pvalue, _ = sm.tsa.coint(S1, S2)
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance_level:
                pairs.append((keys[i], keys[j]))

    return score_matrix, pvalue_matrix, pairs

scores, pvalues, pairs = find_cointegrated_pairs(train_data)
print("Cointegrated pairs using Engle-Granger method:")
for pair in pairs:
    print(pair)

# Johansen测试
def johansen_test(data, significance_level=0.05):
    data = data.dropna(axis=1, how='all')
    if data.shape[1] == 0:
        print("No columns with sufficient data for Johansen test.")
        return None, None

    data = data.dropna()
    data = (data - data.mean()) / data.std()
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    if np.isnan(data).all(axis=0).any():
        data = data[:, ~np.isnan(data).all(axis=0)]
    sufficient_data_columns = np.sum(~np.isnan(data), axis=0) > min_non_missing_values
    data = data[:, sufficient_data_columns]
    if data.shape[1] == 0:
        print("No columns with sufficient non-missing data for Johansen test.")
        return None, None

    nobs = len(data)
    maxlags = min(10, nobs // 2 - 1)
    
    try:
        results = coint_johansen(data, det_order=0, k_ar_diff=maxlags)
        trace_stat = results.lr1
        crit_value = results.cvt[:, 1]  # 5% critical value
        cointegrated_pairs = []

        for i in range(len(trace_stat)):
            if trace_stat[i] > crit_value[i]:
                cointegrated_pairs.append((trace_stat[i], crit_value[i]))

        cointegration_vectors = results.evec
        return results, cointegration_vectors
    except np.linalg.LinAlgError:
        print("Singular matrix error encountered during Johansen test.")
        return None, None

johansen_result, cointegration_vectors = johansen_test(train_data)
if johansen_result is not None:
    print("Cointegrated pairs using Johansen method:")
    print("Cointegration vectors and corresponding stocks:")
    cointegrated_pairs = []
    for i in range(cointegration_vectors.shape[1]):
        vector = cointegration_vectors[:, i]
        stocks = train_data.columns
        pair = list(zip(stocks, vector))
        cointegrated_pairs.append(pair)
        print(f"Cointegration vector {i}:")
        for stock, coeff in pair:
            print(f"{stock}: {coeff}")
        print()

# 选择最优的协整对
def perform_cointegration_test(data):
    coint_test = coint_johansen(data, 0, 1)
    coint_vectors = coint_test.evec
    return coint_vectors

def prepare_cointegrated_pairs(data):
    cointegration_vectors = perform_cointegration_test(data)
    cointegrated_pairs = []
    for i in range(cointegration_vectors.shape[1]):
        vector = cointegration_vectors[:, i]
        pair = (list(data.columns), vector)
        cointegrated_pairs.append(pair)
    return cointegrated_pairs

def select_best_pair(cointegrated_pairs, data):
    best_pair = None
    best_score = 0

    for pair in cointegrated_pairs:
        stock_symbols = pair[0]
        cointegration_vector = pair[1]

        score = np.abs(cointegration_vector).sum()

        if score > best_score:
            best_pair = (stock_symbols, cointegration_vector)
            best_score = score

    return best_pair

# 创建数据索引
dates = pd.date_range(start_date_train, periods=len(train_data), freq='D')
imputer = SimpleImputer(strategy='mean')
train_data_values = imputer.fit_transform(train_data)
train_data = pd.DataFrame(train_data_values, index=dates, columns=train_data.columns)

cointegrated_pairs = prepare_cointegrated_pairs(train_data)
best_pair = select_best_pair(cointegrated_pairs, train_data)

# 打印最佳协整对及其系数
print("Best cointegrated pair and their coefficients:")
best_stocks, best_coefficients = best_pair
for stock, coefficient in zip(best_stocks, best_coefficients):
    print(f"{stock}: {coefficient}")

# 构建套利价差
def construct_spread(data, cointegration_vector):
    data_filled = data.ffill().bfill().astype(float)
    cointegration_vector = np.array(cointegration_vector)
    spread = data_filled.dot(cointegration_vector)
    return pd.Series(spread, index=data_filled.index)

# 生成交易信号
def generate_trading_signals(spread, window, z_score_threshold=2.5, position_size=0.01, stop_loss=0.01, take_profit=0.1):
    signals = pd.DataFrame(index=spread.index)
    signals['spread'] = spread
    signals['signal'] = 0
    signals['position'] = 0.0
    signals['stop_loss'] = 0.0
    signals['take_profit'] = 0.0

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std
    signals['z_score'] = z_score

    signals['signal'] = np.where(z_score > z_score_threshold, -position_size, 
                                 np.where(z_score < -z_score_threshold, position_size, 0))

    for i in range(1, len(signals)):
        if signals.iloc[i]['signal'] != 0:
            if signals.iloc[i - 1]['position'] == 0:
                signals.at[signals.index[i], 'stop_loss'] = float(spread[i] * (1 - stop_loss))
                signals.at[signals.index[i], 'take_profit'] = float(spread[i] * (1 + take_profit))
                signals.at[signals.index[i], 'position'] = float(signals.iloc[i]['signal'])
            elif (spread[i] <= signals.iloc[i - 1]['stop_loss']) or (spread[i] >= signals.iloc[i - 1]['take_profit']):
                signals.at[signals.index[i], 'position'] = 0.0
            else:
                signals.at[signals.index[i], 'position'] = signals.iloc[i - 1]['position']
        else:
            signals.at[signals.index[i], 'position'] = signals.iloc[i - 1]['position']

    signals['positions'] = signals['position'].cumsum()

    return signals

# 计算PNL
def calculate_pnl(data, signals, initial_capital, transaction_cost=0.001):
    log_returns = np.log(data).diff()
    strategy_returns = log_returns.mul(signals['positions'].shift(1), axis=0)
    trades = signals['positions'].diff().abs()
    costs = trades * transaction_cost
    pnl = (strategy_returns.sum(axis=1) - costs) * initial_capital
    return pnl

# 使用测试数据构建套利价差
spread = construct_spread(test_data, best_pair[1])

# 生成交易信号
best_window = 20  # 调整窗口期
best_z_score_threshold = 1.5  # Z分数阈值
signals = generate_trading_signals(spread, best_window, best_z_score_threshold, position_size=0.01, stop_loss=0.01, take_profit=0.1)

# 计算PNL
pnl = calculate_pnl(test_data, signals, initial_capital)

# 分配初始资本并计算累计PNL
allocated_pnl = pnl.cumsum()

# 添加初始金额
allocated_pnl += initial_capital

# 绘制PNL
plt.figure(figsize=(12, 6))
plt.plot(allocated_pnl, label='PNL')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
plt.title('PNL with Initial Capital of 1000')
plt.xlabel('Date')
plt.ylabel('PNL')
plt.legend()
plt.grid(True)
plt.show()

daily_returns = pnl / initial_capital
annual_return = np.mean(daily_returns) * 252
annual_volatility = np.std(daily_returns) * np.sqrt(252)
risk_free_rate = 0  
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

# 计算摘要统计信息
std_pnl = allocated_pnl.std()
downside_std = allocated_pnl[allocated_pnl < 0].std()
drawdown = allocated_pnl - allocated_pnl.cummax()
max_drawdown = drawdown.min()
daily_returns = pnl / initial_capital

# Calculate annual volatility
annual_volatility = np.std(daily_returns) * np.sqrt(252) 
# 打印摘要统计信息
print(f"Standard Deviation of PNL: {std_pnl:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}")
print(f"Annual_volatility:{annual_volatility:.2f}")
# 添加打印每天PNL的代码
daily_pnl = pnl.cumsum() + initial_capital

# 打印每天的PNL
print("Daily PNL values:")
print(daily_pnl)

# 绘制每天PNL的图表
plt.figure(figsize=(12, 6))
plt.plot(daily_pnl, label='Daily PNL')
plt.title('Daily PNL')
plt.xlabel('Date')
plt.ylabel('PNL')
plt.legend()
plt.grid(True)
plt.show()

# 将每日PNL保存为Excel文件
daily_pnl.to_excel('daily_pnl.xlsx', sheet_name='Daily PNL')

if __name__ == '__main__':
    pass
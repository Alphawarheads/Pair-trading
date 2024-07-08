import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

from scipy.optimize import minimize



class PairTrading:
    def __init__(self, sh):
        self.sh = sh

    def SSD(self, priceX, priceY):
        returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (returnX + 1).cumprod()
        standardY = (returnY + 1).cumprod()
        SSD = np.sum((standardY - standardX) ** 2)
        return SSD

    def findPair(self, sh, formStart, formEnd):
        sh_form = sh[formStart:formEnd]
        lst = list(sh_form.columns)
        d = dict()
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                P_zhonghang_f = sh_form[lst[i]]
                P_pufa_f = sh_form[lst[j]]
                dis = self.SSD(P_zhonghang_f, P_pufa_f)
                d[lst[i] + '-' + lst[j]] = dis

        # least ssd, rank top 10
        d_sort = sorted(d.items(), key=lambda x: x[1])
        return d_sort[:20]

    def SSD_Spread(self, priceX, priceY):
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        retx = priceX.diff()[1:]
        rety = priceY.diff()[1:]
        standardX = (1 + retx).cumprod()
        standardY = (1 + rety).cumprod()
        spread = standardY - standardX
        return spread

    def Jump_Spread(self, priceX, priceY):
        spread = np.log(priceX / priceX.iloc[0]) - np.log(priceY / priceY.iloc[0])
        return spread

    def showHistory(self, P_priceX_f, P_priceY_f):
        plt.figure(figsize=(10, 4))
        ax = plt.subplot()
        ax.plot(P_priceX_f, label='X')
        ax.plot(P_priceY_f, label='Y')
        plt.title('Closing Prices', fontsize=15)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Closing Price', fontsize=14)
        ax.legend()
        plt.show()

    def showReturn(self, P_priceX_f, P_priceY_f):
        return_priceX = (P_priceX_f - P_priceX_f.shift(1)) / P_priceX_f.shift(1)[1:]
        return_priceY = (P_priceY_f - P_priceY_f.shift(1)) / P_priceY_f.shift(1)[1:]

        plt.figure(figsize=(10, 4))
        ax = plt.subplot()
        ax.plot(return_priceX, label='P_priceX_f')
        ax.plot(return_priceY, label='P_priceY_f')
        plt.title('Return Rates', fontsize=15)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Return Rates', fontsize=14)
        ax.legend()
        plt.show()

    def showCumReturns(self, P_priceX_f, P_priceY_f):
        return_priceX = (P_priceX_f - P_priceX_f.shift(1)) / P_priceX_f.shift(1)[1:]
        return_priceY = (P_priceY_f - P_priceY_f.shift(1)) / P_priceY_f.shift(1)[1:]

        cum_return_priceX = (1 + return_priceX).cumprod()
        cum_return_priceY = (1 + return_priceY).cumprod()

        plt.figure(figsize=(10, 4))
        ax = plt.subplot()
        ax.plot(cum_return_priceX, label='return_priceX')
        ax.plot(cum_return_priceY, label='return_priceY')
        plt.title('Cumulative Return', fontsize=15)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return Rates', fontsize=14)
        ax.legend()
        plt.show()

    def SSD_Cal_Bound(self, priceX, priceY, width=2):
        spread = self.SSD_Spread(priceX, priceY)
        mu = np.mean(spread)
        sd = np.std(spread)
        UpperBound = mu + width * sd
        LowerBound = mu - width * sd
        return UpperBound, LowerBound

    def ADF_check(self, priceX, priceY):
        result_X = adfuller(priceX)
        result_Y = adfuller(priceY)
        print(result_X)
        print(result_Y)

    def ADF_diff_check(self, priceX, priceY):
        priceX = np.diff(priceX)
        priceY = np.diff(priceY)
        result_X = adfuller(priceX)
        result_Y = adfuller(priceY)
        print(result_X)
        print(result_Y)

    def coint_check(self, priceX, priceY):
        print(coint(priceX, priceY))

    def plotBands(self, mu, sd):
        plt.figure(figsize=(10, 6))
        SSD_spread_trade.plot()
        plt.title('Cointegrated Pairs (Trading Period)', loc='center', fontsize=16)
        plt.axhline(y=mu, color='black')
        plt.axhline(y=mu + 0.2 * sd, color='blue', ls='-', lw=2)
        plt.axhline(y=mu - 0.2 * sd, color='blue', ls='-', lw=2)
        plt.axhline(y=mu + 1.5 * sd, color='green', ls='--', lw=2.5)
        plt.axhline(y=mu - 1.5 * sd, color='green', ls='--', lw=2.5)
        plt.axhline(y=mu + 3.0 * sd, color='red', ls='-.', lw=3)
        plt.axhline(y=mu - 3.0 * sd, color='red', ls='-.', lw=3)
        plt.show()

    def TradeSig(self, prcLevel):
        n = len(prcLevel)
        signal = np.zeros(n)

        for i in range(1, n):
            if prcLevel.iloc[i - 1] == 1 and prcLevel.iloc[i] == 2:
                signal[i] = -2
            elif prcLevel.iloc[i - 1] == 1 and prcLevel.iloc[i] == 0:
                signal[i] = 2
            elif prcLevel.iloc[i - 1] == 2 and prcLevel.iloc[i] == 3:
                signal[i] = 3
            elif prcLevel.iloc[i - 1] == -1 and prcLevel.iloc[i] == -2:
                signal[i] = 1
            elif prcLevel.iloc[i - 1] == -1 and prcLevel.iloc[i] == 0:
                signal[i] = -1
            elif prcLevel.iloc[i - 1] == -2 and prcLevel.iloc[i] == -3:
                signal[i] = -3
        return signal

    def TradeSim(self, priceX, priceY, position, allocation):
        n = len(position)
        size = 1000 * allocation
        beta = 1  # 确定交易头寸：等权重；delta对冲；资金比例
        shareY = size * position  # shareY为兴业银行标的数量
        shareX = [(-beta) * shareY.iloc[0] * priceY.iloc[0] / priceX.iloc[0]]
        cash = [2000 * allocation]  # 注意保证金（以20%为例，即5倍杠杆）与配对交易的市值（size*标的价格）

        for i in range(1, n):
            # 初始化当前时间步的股票数量和现金余额为前一个时间步的值
            shareX.append(shareX[i - 1])
            cash.append(cash[i - 1])

            # 根据头寸变化调整持有的股票数量和现金余额
            if position.iloc[i - 1] == 0 and position.iloc[i] == 1:
                shareX[i] = (-beta) * shareY.iloc[i] * priceY.iloc[i] / priceX.iloc[i]
                cash[i] = cash[i - 1] - (shareY.iloc[i] * priceY.iloc[i] + shareX[i] * priceX.iloc[i])
            elif position.iloc[i - 1] == 0 and position.iloc[i] == -1:
                shareX[i] = (-beta) * shareY.iloc[i] * priceY.iloc[i] / priceX.iloc[i]
                cash[i] = cash[i - 1] - (shareY.iloc[i] * priceY.iloc[i] + shareX[i] * priceX.iloc[i])
            elif position.iloc[i - 1] == 1 and position.iloc[i] == -1:
                shareX[i] = (-beta) * shareY.iloc[i] * priceY.iloc[i] / priceX.iloc[i]
                cash[i] = cash[i - 1] + shareX[i - 1] * priceX.iloc[i] - shareX[i] * priceX.iloc[i]
            elif position.iloc[i - 1] == -1 and position.iloc[i] == 1:
                shareX[i] = (-beta) * shareY.iloc[i] * priceY.iloc[i] / priceX.iloc[i]
                cash[i] = cash[i - 1] + shareX[i - 1] * priceX.iloc[i] - shareX[i] * priceX.iloc[i]

        # 转换为 Pandas 数据结构并计算资产总值
        cash = pd.Series(cash, index=position.index)
        shareY = pd.Series(shareY, index=position.index)
        shareX = pd.Series(shareX, index=position.index)
        asset = cash + shareY * priceY + shareX * priceX
        account = pd.DataFrame({'Position': position, 'ShareY': shareY, 'ShareX': shareX, 'Cash': cash, 'Asset': asset})
        return account

    def compute_summary_statistics(self, account):
        # Ensure the index is in datetime format
        account.index = pd.to_datetime(account.index)
        account = account.sort_index()

        # Drop any rows where 'Asset' is NaN
        account = account.dropna(subset=['Asset'])

        returns = account['Asset'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        # Annualization factor assuming daily returns
        annualization_factor = np.sqrt(252)
        sharpe_ratio = (mean_return / std_return) * annualization_factor
        ir = mean_return / std_return
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        # Annual Returns
        if len(account) > 1 and account['Asset'].iloc[0] != 0:
            total_period = (account.index[-1] - account.index[0]).days / 365.25
            annual_return = (account['Asset'].iloc[-1] / account['Asset'].iloc[0]) ** (1 / total_period) - 1
        else:
            annual_return = np.nan  # or handle appropriately if there's an issue
        return {
            'Information Ratio (IR)': ir,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Annual Return': annual_return
        }

    def strategy_formulation(self, formStart, formEnd):
        pairs = self.findPair(self.sh, formStart, formEnd)
        self.pairs = pairs

    def backtest_pair(self, pair, tradStart, tradEnd, allocation):
        pa = pair.split("-")
        priceX = self.sh[pa[0]]
        priceY = self.sh[pa[1]]
        price_priceX_trade = priceX[tradStart:tradEnd]
        price_priceY_trade = priceY[tradStart:tradEnd]
        SSD_spread_trade = self.SSD_Spread(price_priceX_trade, price_priceY_trade)
        mu = np.mean(SSD_spread_trade)
        sd = np.std(SSD_spread_trade)

        self.level = sorted((
            float('-inf'), mu - 3.0 * sd, mu - 1.5 * sd, mu - 0.2 * sd, mu + 0.2 * sd, mu + 1.5 * sd, mu + 3.0 * sd,
            float('inf')
        ))

        prcLevel = pd.cut(SSD_spread_trade, self.level, labels=False) - 3
        signal = self.TradeSig(prcLevel)
        position = [signal[0]]
        ns = len(signal)
        for i in range(1, ns):
            position.append(position[-1])
            if signal[i] == 1:
                position[i] = 1
            elif signal[i] == -2:
                position[i] = -1
            elif signal[i] == -1 and position[i - 1] == 1:
                position[i] = 0
            elif signal[i] == 2 and position[i - 1] == -1:
                position[i] = 0
            elif signal[i] == 3:
                position[i] = 0
            elif signal[i] == -3:
                position[i] = 0
        position = pd.Series(position, index=SSD_spread_trade.index)
        account = self.TradeSim(price_priceX_trade, price_priceY_trade, position, allocation)
        return account

    def optimize_allocation(self, returns):
        def portfolio_variance(weights, cov_matrix):
            return weights.T @ cov_matrix @ weights

        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets, ]

        optimized = minimize(portfolio_variance, initial_guess, args=cov_matrix,
                             method='SLSQP', bounds=bounds, constraints=constraints)

        return optimized.x

    def aggregate_backtests(self, tradStart, tradEnd):
        all_accounts = []
        pair_names = []
        for pair, _ in self.pairs:
            print(f"Backtesting pair: {pair}")
            account = self.backtest_pair(pair, tradStart, tradEnd, allocation=1.0)
            all_accounts.append(account['Asset'])
            pair_names.append(pair)

            # Print individual pair returns
            print(f"Returns for {pair}:")
            print(account['Asset'])

        combined_account = pd.concat(all_accounts, axis=1)
        combined_account.columns = pair_names

        # Calculate returns for optimization
        returns = combined_account.pct_change(fill_method=None).dropna()

        # Optimize allocations using Markowitz Portfolio Theory
        optimal_allocations = self.optimize_allocation(returns)

        print(f"Optimal Allocations: {optimal_allocations}")

        combined_account['Total'] = combined_account.dot(optimal_allocations)

        # Save to CSV
        combined_account.to_csv('pair_trading_results.csv')

        # Plot combined account
        combined_account['Total'].plot(style='--', color='blue', figsize=(10, 6))
        plt.title('Combined Account', loc='center', fontsize=16)
        plt.show()

        # Calculate and print summary statistics for the combined account
        summary_stats = self.compute_summary_statistics(pd.DataFrame({'Asset': combined_account['Total']}))
        print("Summary Statistics for Combined Account:")
        for key, value in summary_stats.items():
            print(f"{key}: {value}")

        return combined_account

    def fill_nan_with_random(self, df):
        for col in df.columns:
            for i in range(1, len(df)):
                if pd.isna(df.iloc[i, df.columns.get_loc(col)]):
                    prev_value = df.iloc[i-1, df.columns.get_loc(col)]
                    random_value = prev_value * (1 + np.random.uniform(-0.01, 0.01))
                    df.iloc[i, df.columns.get_loc(col)] = random_value
        return df


if __name__ == '__main__':
    sh = pd.read_csv('data.csv', index_col='0')
    pt = PairTrading(sh)
    pt.sh = pt.fill_nan_with_random(pt.sh)
    formStart = '2014-01-01'
    formEnd = '2022-01-01'
    pt.strategy_formulation(formStart=formStart, formEnd=formEnd)
    tradStart = '2022-01-02'
    tradEnd = '2024-06-30'
    pt.aggregate_backtests(tradStart, tradEnd)
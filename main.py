import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request as urllib2
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import baostock as bs
from datetime import datetime, timedelta
from dateutil import parser


class PairTrading:

    def SSD(self, priceX, priceY):
        returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (returnX + 1).cumprod()
        standardY = (returnY + 1).cumprod()
        SSD = np.sum((standardY - standardX) ** 2)
        return (SSD)
    def findPair(self, sh, fromStart, fromEnd):

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
        # print(d_sort)
        return d_sort[:10]

    def SSD_Spread(self, priceX, priceY):
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        retx = priceX.diff()[1:]
        rety = priceY.diff()[1:]
        standardX = (1 + retx).cumprod()
        standardY = (1 + rety).cumprod()
        spread = standardY - standardX
        return (spread)

    def showHistory(self,P_priceX_f,P_priceY_f):
        plt.figure(figsize=(10, 4))
        ax = plt.subplot()

        ax.plot(P_priceX_f, label='X')
        ax.plot(P_priceY_f, label='Y')
        plt.title('closing prives', fontsize=15)
        ax.set_xlabel('date', fontsize=14)
        ax.set_ylabel('closing price', fontsize=14)
        ax.legend()

        plt.show()
    def showReturn(self,P_priceX_f,P_priceY_f):
        return_priceX = (P_priceX_f - P_priceX_f.shift(1)) / P_priceX_f.shift(1)[1:]
        return_priceY = (P_priceY_f - P_priceY_f.shift(1)) / P_priceY_f.shift(1)[1:]

        plt.figure(figsize=(10, 4))
        ax = plt.subplot()

        ax.plot(return_priceX, label='P_priceX_f')
        ax.plot(return_priceY, label='P_priceY_f')
        plt.title('return rates', fontsize=15)
        ax.set_xlabel('date', fontsize=14)
        ax.set_ylabel('return rates', fontsize=14)
        ax.legend()

        plt.show()
    def showCumReturns(self,P_priceX_f,P_priceY_f):

        return_priceX = (P_priceX_f - P_priceX_f.shift(1)) / P_priceX_f.shift(1)[1:]
        return_priceY = (P_priceY_f - P_priceY_f.shift(1)) / P_priceY_f.shift(1)[1:]

        cum_return_priceX = (1 + return_priceX).cumprod()
        cum_return_priceY = (1 + return_priceY).cumprod()

        plt.figure(figsize=(10, 4))
        ax = plt.subplot()

        ax.plot(cum_return_priceX, label='return_priceX')
        ax.plot(cum_return_priceY, label='return_priceY')
        plt.title('cumulative return', fontsize=15)
        ax.set_xlabel('date', fontsize=14)
        ax.set_ylabel('cumulative return rates', fontsize=14)
        ax.legend()

        plt.show()
    def SSD_Cal_Bound(self, priceX, priceY, width=2):
        spread = self.SSD_Spread(priceX, priceY)
        mu = np.mean(spread)
        sd = np.std(spread)
        UpperBound = mu + width * sd
        LowerBound = mu - width * sd
        return (UpperBound, LowerBound)

    def ADF_check(self,priceX, priceY):

        result_X = adfuller(priceX)
        result_Y = adfuller(priceY)
        print(result_X)
        print(result_Y)

    def ADF_diff_check(self,priceX, priceY):

        priceX = np.diff(priceX)
        priceY = np.diff(priceY)
        result_X = adfuller(priceX)
        result_Y = adfuller(priceY)
        print(result_X)
        print(result_Y)

    def coint_check(self,priceX, priceY):
        print(coint(priceX, priceY))
    def plotBands(self,mu,sd):
        plt.figure(figsize=(10, 6))

        SSD_spread_trade.plot()

        plt.title('Cointegrated pairs(Trading period)', loc='center', fontsize=16)

        plt.axhline(y=mu, color='black')

        plt.axhline(y=mu + 0.2 * sd, color='blue', ls='-', lw=2)
        plt.axhline(y=mu - 0.2 * sd, color='blue', ls='-', lw=2)
        plt.axhline(y=mu + 1.5 * sd, color='green', ls='--', lw=2.5)
        plt.axhline(y=mu - 1.5 * sd, color='green', ls='--', lw=2.5)
        plt.axhline(y=mu + 3.0 * sd, color='red', ls='-.', lw=3)
        plt.axhline(y=mu - 3.0 * sd, color='red', ls='-.', lw=3)

        plt.show()
    def TradeSig(self,prcLevel):
        n = len(prcLevel)
        signal = np.zeros(n)

        for i in range(1, n):
            if prcLevel[i - 1] == 1 and prcLevel[i] == 2:
                signal[i] = -2
            elif prcLevel[i - 1] == 1 and prcLevel[i] == 0:
                signal[i] = 2
            elif prcLevel[i - 1] == 2 and prcLevel[i] == 3:
                signal[i] = 3
            elif prcLevel[i - 1] == -1 and prcLevel[i] == -2:
                signal[i] = 1
            elif prcLevel[i - 1] == -1 and prcLevel[i] == 0:
                signal[i] = -1
            elif prcLevel[i - 1] == -2 and prcLevel[i] == -3:
                signal[i] = -3
        return (signal)

    def TradeSim(self,priceX, priceY, position):
        n = len(position)
        size = 1000
        beta = 1  # Determine trading positions: Equal weighting; Delta hedging; Capital ratio
        shareY = size * position
        shareX = [(-beta) * shareY[0] * priceY[0] / priceX[0]]
        cash = [2000]  # margin (for example, 20%, 5 times leverage) and the market value of the paired trade (size * price of the underlying asset)"

        for i in range(1, n):
            shareX.append(shareX[i - 1])
            cash.append(cash[i - 1])
            if position[i - 1] == 0 and position[i] == 1:
                shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
                cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
            elif position[i - 1] == 0 and position[i] == -1:
                shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
                cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
            elif position[i - 1] == 1 and position[i] == 0:
                shareX[i] = 0
                cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
            elif position[i - 1] == -1 and position[i] == 0:
                shareX[i] = 0
                cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])

        cash = pd.Series(cash, index=position.index)
        shareY = pd.Series(shareY, index=position.index)
        shareX = pd.Series(shareX, index=position.index)
        asset = cash + shareY * priceY + shareX * priceX
        account = pd.DataFrame({'Position': position, 'ShareY': shareY, 'ShareX': shareX, 'Cash': cash, 'Asset': asset})
        return (account)

#
    def compute_summary_statistics(self, account):

        returns = account['Asset'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()

        # Annualization factor assuming daily returns
        annualization_factor = np.sqrt(252)#although in finding alpha(or investments), a 256 was given

        sharpe_ratio = (mean_return / std_return) * annualization_factor
        ir = mean_return / std_return

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'Information Ratio (IR)': ir,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown
        }





if __name__ == '__main__':
    pt = PairTrading()
    sh = pd.read_csv('sh50.csv', index_col='Trddt')
    sh.index = pd.to_datetime(sh.index)
    # print(sh.head())
    # print(sh.tail)

#strategy fomulation
    formStart = '2014-01-01'
    formEnd = '2014-12-31'
    pair=pt.findPair(sh, formStart, formEnd)[0]
    print(pair)
    P_priceX = sh['600015']
    P_priceY = sh['601166']

    #check visually
    pt.showHistory(P_priceX,P_priceY)
    pt.showReturn(P_priceX,P_priceY)
    pt.showCumReturns(P_priceX,P_priceY)

    price_priceX_form = P_priceX[formStart:formEnd]
    price_priceY_form = P_priceY[formStart:formEnd]

#constructing bands
    SSD_spread_form = pt.SSD_Spread(price_priceX_form, price_priceY_form)
    mu = np.mean(SSD_spread_form)
    sd = np.std(SSD_spread_form)

#trading simulation and backtesting
    tradStart = '2015-01-01'
    tradEnd = '2015-12-31'

    price_priceX_trade = P_priceX[tradStart:tradEnd]
    price_priceY_trade = P_priceY[tradStart:tradEnd]

    SSD_spread_trade = pt.SSD_Spread(price_priceX_trade, price_priceY_trade)
    SSD_spread_trade.head()

    pt.plotBands(mu, sd)

    level = (float('-inf'), mu - 3.0 * sd, mu - 1.5 * sd, mu - 0.2 * sd, mu + 0.2 * sd, mu + 1.5 * sd, mu + 3.0 * sd,
             float('inf'))

    prcLevel = pd.cut(SSD_spread_trade, level, labels=False) - 3
    signal = pt.TradeSig(prcLevel)

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

    account = pt.TradeSim(price_priceX_trade, price_priceY_trade, position)


    account.iloc[:, [1, 2, 3, 4]].plot(style=['--', '--', '-', ':'], color=['red', 'blue', 'yellow', 'green'],
                                       figsize=(10, 6))

    plt.title('Account', loc='center', fontsize=16)

    plt.show()

    summary_stats = pt.compute_summary_statistics(account)
    for i in summary_stats: print(i+':', summary_stats[i])
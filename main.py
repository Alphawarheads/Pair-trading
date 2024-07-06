import pandas as pd
from Zhenshan import PairTrading as pair
# from Yanchen import PairTrading as pair
# from Qizheng import PairTrading as pair



if __name__ == '__main__':
    sh = pd.read_csv('data.csv', index_col='0')
    pt = pair(sh)
    formStart = '2017-01-01'
    formEnd = '2022-12-31'
    pt.strategy_formulation(formStart=formStart, formEnd=formEnd)
    tradStart = '2017-01-01'
    tradEnd = '2022-12-31'
    pt.aggregate_backtests(tradStart, tradEnd)

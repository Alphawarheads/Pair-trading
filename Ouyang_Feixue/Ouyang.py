import time
from multiprocessing import Queue, Process
import numpy as np
import pandas as pd
from itertools import combinations
from numba import jit, prange
import gc
import baostock as bs
from statsmodels.tsa.stattools import coint
from tqdm import tqdm


@jit(nopython=True)
def estimate_fou_params(spread_series):
    a = np.array([0.482962913144534 / (2 ** 0.5), -0.836516303737808 / (2 ** 0.5), 0.224143868042013 / (2 ** 0.5),
                  0.12940952255126 / (2 ** 0.5)])
    a2 = np.array(
        [0.482962913144534 / (2 ** 0.5), 0, -0.836516303737808 / (2 ** 0.5), 0, 0.224143868042013 / (2 ** 0.5), 0,
         0.12940952255126 / (2 ** 0.5)])

    def Vn_a(series, filter_a):
        filtered_series = np.convolve(series, filter_a, mode='valid')
        return np.sum(filtered_series ** 2)

    Vn_a_val = Vn_a(spread_series, a)
    Vn_a2_val = Vn_a(spread_series, a2)

    if Vn_a_val == 0 or Vn_a2_val == 0:
        raise ValueError("Vn_a_val or Vn_a2_val is 0, please check your spread_series data.")

    H = 0.5 * np.log2(Vn_a2_val / Vn_a_val)

    C_sum = 0.0
    for i in range(len(a)):
        for j in range(len(a)):
            C_sum += a[i] * a[j] * abs(i - j) ** (2 * H)

    sigma = np.sqrt((-2 * Vn_a_val) / C_sum)

    spread_mean = np.mean(spread_series)

    Y = spread_series - spread_mean

    Y2 = np.mean(Y ** 2)

    gamma_val = np.math.gamma(2 * H + 1)

    sigma_squared_gamma = sigma ** 2 * gamma_val

    theta_base = 2 * Y2 / sigma_squared_gamma

    if theta_base <= 0:
        raise ValueError("Theta base is non-positive, cannot compute power.")

    theta = np.power(theta_base, -1 / (2 * H))

    mu = np.mean(spread_series) / theta

    return float(H), float(sigma), float(theta), float(mu)


@jit(nopython=True, parallel=True)
def calculate_spread_params_parallel(spread, rolling_window_size, initial_H, initial_k):
    n = len(spread)
    k = np.full(n, initial_k)

    for i in prange(rolling_window_size, n):
        rolling_window_data = spread[i - rolling_window_size:i]
        H, sigma, _, mu = estimate_fou_params(rolling_window_data)

        k[i] = initial_k * (H / initial_H)

    return k

def generate_signals_with_dynamic_thresholds(prices1, prices2, open_times, initial_mu, initial_sigma, crypto1, crypto2, initial_H,
                                             initial_k,initial_price1, initial_price2):
    """
    生成交易信号，使用动态调整的阈值
    """
    # 将输入序列转换为 NumPy 数组
    prices1 = np.array(prices1)
    prices2 = np.array(prices2)
    open_times = np.array(open_times)

    log_prices1 = np.log(prices1 / initial_price1)
    log_prices2 = np.log(prices2 / initial_price2)
    spread = log_prices1 - log_prices2

    rolling_window_size = 48 * 20
    spread_std = np.full(len(spread), initial_sigma)
    spread_mean = np.full(len(spread), initial_mu)

    k = calculate_spread_params_parallel(spread, rolling_window_size, initial_H, initial_k)

    upper_threshold = spread_mean + k * spread_std
    lower_threshold = spread_mean - k * spread_std

    signals = []
    Pair = f'{crypto1}-{crypto2}'
    long_position = False
    short_position = False

    for index in range(rolling_window_size, len(spread)):
        is_last_day = index == len(spread) - 1
        current_spread = spread[index]

        if current_spread > upper_threshold[index] and not short_position and not long_position:
            signals.append({
                'Time': open_times[index],
                'Signal': 'Sell',
                'Pair': Pair,
            })
            short_position = True

        if current_spread < lower_threshold[index] and not long_position and not short_position:
            signals.append({
                'Time': open_times[index],
                'Signal': 'Buy',
                'Pair': Pair,
            })
            long_position = True

        if long_position and (current_spread >= spread_mean[index] or is_last_day):
            signals.append({
                'Time': open_times[index],
                'Signal': 'Close-Buy',
                'Pair': Pair,
            })
            long_position = False

        if short_position and (current_spread <= spread_mean[index] or is_last_day):
            signals.append({
                'Time': open_times[index],
                'Signal': 'Close-Sell',
                'Pair': Pair,
            })
            short_position = False

    gc.collect()

    return signals


def query_with_retry(query_func, *args, max_retries=100, **kwargs):
    retries = 0
    bs.login()
    while retries < max_retries:
        try:
            rs = query_func(*args, **kwargs)
            if rs.error_code != '0':
                raise Exception(f"Error code {rs.error_code}: {rs.error_msg}")
            bs.logout()
            return rs
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            retries += 1
    raise Exception("Max retries exceeded. Failed to fetch data.")


def get_price_with_retry(stock, start_time, end_time, max_retries=100):
    retries = 0
    while retries < max_retries:
        try:
            rs = bs.query_history_k_data_plus(stock,
                                              "time,code,close",
                                              start_date=start_time, end_date=end_time,
                                              frequency="5", adjustflag="2")
            if rs.error_code != '0':
                raise Exception(f"Error code {rs.error_code}: {rs.error_msg}")

            data_list = []
            while rs.next():
                # 获取一条记录，将记录合并在一起
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=rs.fields)
            result['close'] = result['close'].astype(float)
            return result
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 1 seconds...")
            time.sleep(1)
            retries += 1
    raise Exception("Max retries exceeded. Failed to fetch data.")


def preprocess_data(queue, window_days=80, step_days=10, rolling_days=20):

    # 获取交易日数据
    rs = query_with_retry(bs.query_trade_dates, start_date='2008-01-01', end_date='2023-12-31')
    data_list = []
    while rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    trading_days = pd.DataFrame(data_list, columns=rs.fields)
    trading_days = trading_days[trading_days['is_trading_day'] == '1'].reset_index(drop=True)

    start_index = 0
    period_days = 120
    while start_index + period_days <= len(trading_days):
        period_start_time = trading_days.iloc[start_index]['calendar_date']
        period_end_time = trading_days.iloc[start_index + period_days - 1]['calendar_date']

        # 获取沪深300成分股
        rs = query_with_retry(bs.query_hs300_stocks, date=period_start_time)
        hs300_stocks = []
        while rs.next():
            # 获取一条记录，将记录合并在一起
            hs300_stocks.append(rs.get_row_data())
        index_component = pd.DataFrame(hs300_stocks, columns=rs.fields)['code']

        period_data_dict = {}
        bs.login()
        for stock in tqdm(index_component):
            period_data = get_price_with_retry(stock=stock, start_time=period_start_time, end_time=period_end_time)
            period_data['time'] = pd.to_datetime(period_data['time'], format='%Y%m%d%H%M%S%f')
            period_data_dict.update({stock: period_data})
        bs.logout()
            # 可以根据需要调整时间

        # 滚动窗口处理
        for start_offset in range(0, period_days - window_days + 1, step_days):
            time_indices = {
                'start_time': pd.to_datetime(trading_days.iloc[start_index + start_offset]['calendar_date']),
                'end_time': pd.to_datetime(
                    trading_days.iloc[start_index + start_offset + window_days]['calendar_date']),
                'future_start_time': pd.to_datetime(
                    trading_days.iloc[start_index + start_offset + window_days - rolling_days - 1]['calendar_date']),
                'future_end_time': pd.to_datetime(
                    trading_days.iloc[start_index + start_offset + window_days + step_days - 1]['calendar_date'])
            }

            window_data_dict = {}
            future_data_dict = {}

            for stock in index_component:
                window_data = period_data_dict[stock][
                    (period_data_dict[stock]['time'] >= time_indices['start_time']) &
                    (period_data_dict[stock]['time'] <= time_indices['end_time'])]
                future_data = period_data_dict[stock][
                    (period_data_dict[stock]['time'] >= time_indices['future_start_time']) &
                    (period_data_dict[stock]['time'] <= time_indices['future_end_time'])]
                window_data_dict.update({stock: window_data})
                future_data_dict.update({stock: future_data})

            all_pairs = combinations(index_component, 2)
            for stock1, stock2 in all_pairs:
                price1 = window_data_dict[stock1]
                price2 = window_data_dict[stock2]
                future_price1 = future_data_dict[stock1]
                future_price2 = future_data_dict[stock2]

                if len(price1) == len(price2) == 1920 * 2 and len(future_price1) == len(future_price2) == 720 * 2:
                    queue.put(
                        (stock1, stock2, price1['close'].values, price2['close'].values, future_price1, future_price2))

            queue.put('flag')
            print(f"Data fetching end at: {time_indices['end_time']}")

            # 清理不需要的数据，释放内存
            del window_data_dict
            del future_data_dict

        start_index += period_days

        # 清理不需要的数据，释放内存
        del period_data_dict
        del index_component

    queue.put(None)

    # 强制进行垃圾回收
    import gc
    gc.collect()


def calculate_rolling_cointegration_for_all_pairs(queue):
    results = []
    waiting_list = []
    future_data_dict = {}

    while True:
        data = queue.get()
        if data is None:
            break

        if data == 'flag':
            if waiting_list:
                waiting_df = pd.DataFrame(waiting_list)
                waiting_df['rank theta'] = waiting_df['theta'].rank(ascending=False)
                waiting_df['rank H'] = waiting_df['H'].rank(ascending=True)
                waiting_df['total score'] = waiting_df['rank theta'] + waiting_df['rank H']
                top_n = 10
                top_n_df = waiting_df.nlargest(top_n, 'total score').reset_index(drop=True)

                for _, row in top_n_df.iterrows():
                    stock1, stock2 = row['Pair'].split('-')
                    future_prices1 = future_data_dict[stock1][0]['close'].values
                    future_prices2 = future_data_dict[stock2][0]['close'].values
                    initial_price1 = future_data_dict[stock1][1]
                    initial_price2 = future_data_dict[stock2][1]
                    open_times = future_data_dict[stock1][0]['time']

                    H = row['H']
                    mu = row['mu']
                    sigma = row['sigma']
                    signals = generate_signals_with_dynamic_thresholds(prices1=future_prices1,prices2= future_prices2,open_times= open_times,
                                                                       initial_mu=mu, initial_sigma=sigma,
                                                                       crypto1=stock1, crypto2=stock2,
                                                                       initial_k=0.7, initial_H=H,initial_price1=initial_price1,initial_price2=initial_price2)
                    results.extend(signals)
                print(f'Signals generated')
                waiting_list = []
                future_data_dict = {}

                del waiting_df, top_n_df
                gc.collect()
            else:
                print(f'No Signals generated')

        else:
            stock1, stock2, price1, price2,future_price1,future_price2 = data
            unpacked_result = process_pair(price1, price2, stock1, stock2)
            if unpacked_result is not None:
                result, initial_price1, initial_price2 = unpacked_result
                waiting_list.append(result)
                future_data_dict.update({stock1:(future_price1,initial_price1),stock2:(future_price2,initial_price2)})

    results_df = pd.DataFrame(results)
    results_df.to_csv('trading_signal_CSI_80.csv')


def process_pair(price1, price2, crypto1, crypto2):
    coint_t, p_value, _ = coint(price1, price2, trend='c', maxlag=1, autolag=None)
    if p_value <0.05:
        log_prices1 = np.log(price1 / price1[0])
        log_prices2 = np.log(price2 / price2[0])
        spread = log_prices1 - log_prices2
        try:
            H, sigma, theta, mu = estimate_fou_params(spread)

            return ({
                'Pair': f'{crypto1}-{crypto2}',
                'theta': theta,
                'H': H,
                'mu': mu,
                'sigma': sigma,
                'crypto1': crypto1,
                'crypto2': crypto2
            },price1[0],price2[0])
        except:
            return None
    else:
        return None
    # log_prices1 = np.log(price1 / price1[0])
    # log_prices2 = np.log(price2 / price2[0])
    # spread = log_prices1 - log_prices2
    # try:
    #     H, sigma, theta, mu = estimate_fou_params(spread)
    #
    #     return {
    #         'Pair': f'{crypto1}-{crypto2}',
    #         'theta': theta,
    #         'H': H,
    #         'mu': mu,
    #         'sigma': sigma,
    #         'crypto1': crypto1,
    #         'crypto2': crypto2
    #     }
    # except:
    #     return None


if __name__ == "__main__":
    queue = Queue(maxsize=10)

    producer = Process(target=preprocess_data, args=(queue,))
    consumer = Process(target=calculate_rolling_cointegration_for_all_pairs, args=(queue,))

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

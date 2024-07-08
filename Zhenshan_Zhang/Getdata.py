import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
def data_collection(from_date, to_date):
    lg = bs.login()

    # gain a list of avaliable stocks
    rs = bs.query_hs300_stocks()
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():

        hs300_stocks.append(rs.get_row_data()[1])

    #creating the first line of
    rs = bs.query_history_k_data_plus(hs300_stocks[0],
            "date,close",
            start_date=from_date, end_date=to_date,
            frequency="d")

    data_list=[]
    date_list=[]

    while (rs.error_code == '0') & rs.next():
        data=rs.get_row_data()

        date_list.append(data[0])
        data_list.append(data[1])

    date = pd.DataFrame(date_list)
    df = pd.DataFrame({hs300_stocks[0].strip("sh.").strip("z."): data_list})
    df = pd.concat((date,df),axis=1)

#collecting the rest of the stocks
    for i in hs300_stocks[1:]:
        rs = bs.query_history_k_data_plus(i,
            "date,close",
            start_date=from_date, end_date=to_date,
            frequency="d")


        data_list=[]
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data()[1])


        d= pd.DataFrame({i.strip("sh.").strip("z."): data_list})
        df = pd.concat((df, d), axis=1)
    df.to_csv("./data.csv", index=False)

    bs.logout()
if __name__ == '__main__':
    data_collection('2014-01-01','2024-06-30')
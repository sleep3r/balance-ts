import pandas as pd


def get_rates() -> pd.DataFrame:
    rates = pd.read_csv("rates.csv", index_col=0)
    rates.columns = ['Date', 'key_rate']
    rates.Date = rates.Date.apply(pd.Timestamp)
    rates.set_index('Date', inplace=True)
    rates['key_rate'] /= 1e4
    return rates

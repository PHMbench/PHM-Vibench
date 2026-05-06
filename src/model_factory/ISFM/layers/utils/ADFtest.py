import pandas as pd
import numpy as np
import os

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError as exc:  # pragma: no cover - optional dependency path
    adfuller = None
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None

try:
    from arch.unitroot import ADF
except ImportError as exc:  # pragma: no cover - optional dependency path
    ADF = None
    _ARCH_IMPORT_ERROR = exc
else:
    _ARCH_IMPORT_ERROR = None


def _require_adfuller():
    if adfuller is None:
        raise ImportError(
            "statsmodels is required for ADFtest.calculate_ADF. "
            "Install it with `uv pip install statsmodels`."
        ) from _STATSMODELS_IMPORT_ERROR
    return adfuller


def _require_arch_adf():
    if ADF is None:
        raise ImportError(
            "arch is required for ADFtest.archADF. "
            "Install it with `uv pip install arch`."
        ) from _ARCH_IMPORT_ERROR
    return ADF

def calculate_ADF(root_path,data_path):
    adfuller_fn = _require_adfuller()
    df_raw = pd.read_csv(os.path.join(root_path,data_path))
    cols = list(df_raw.columns)
    cols.remove('date')
    df_raw = df_raw[cols]
    adf_list = []
    for i in cols:
        df_data = df_raw[i]
        adf = adfuller_fn(df_data, maxlag = 1)
        print(adf)
        adf_list.append(adf)
    return np.array(adf_list)

def calculate_target_ADF(root_path,data_path,target='OT'):
    adfuller_fn = _require_adfuller()
    df_raw = pd.read_csv(os.path.join(root_path,data_path))
    target_cols = target.split(',')
    # df_data = df_raw[target]
    df_raw = df_raw[target_cols]
    adf_list = []
    for i in target_cols:
        df_data = df_raw[i]
        adf = adfuller_fn(df_data, maxlag = 1)
        # print(adf)
        adf_list.append(adf)
    return np.array(adf_list)

def archADF(root_path, data_path):
    adf_cls = _require_arch_adf()
    df = pd.read_csv(os.path.join(root_path,data_path))
    cols = df.columns[1:]
    stats = 0
    for target_col in cols:
        series = df[target_col].values
        adf = adf_cls(series)
        stat = adf.stat
        stats += stat
    return stats/len(cols)

if __name__ == '__main__':

    # * Exchange - result: -1.902402344564288 | report: -1.889
    ADFmetric = archADF(root_path="./dataset/exchange_rate/",data_path="exchange_rate.csv")
    print("Exchange ADF metric", ADFmetric)

    # * Illness - result: -5.33416661870624 | report: -5.406
    ADFmetric = archADF(root_path="./dataset/illness/",data_path="national_illness.csv") 
    print("Illness ADF metric", ADFmetric)

    # * ETTm2 - result: -5.663628743471695 | report: -6.225
    ADFmetric = archADF(root_path="./dataset/ETT-small/",data_path="ETTm2.csv")
    print("ETTm2 ADF metric", ADFmetric)

    # * Electricity - result: -8.44485821939281 | report: -8.483
    ADFmetric = archADF(root_path="./dataset/electricity/",data_path="electricity.csv")
    print("Electricity ADF metric", ADFmetric)

    # * Traffic - result: -15.020978067839014 | report: -15.046
    ADFmetric = archADF(root_path="./dataset/traffic/",data_path="traffic.csv")
    print("Traffic ADF metric", ADFmetric)

    # * Weather - result: -26.681433085204866 | report: -26.661
    ADFmetric = archADF(root_path="./dataset/weather/",data_path="weather.csv")
    print("Weather ADF metric", ADFmetric)


    # print(ADFmetric)

    # mean_ADFmetric = ADFmetric[:,0].mean()
    # print(mean_ADFmetric)

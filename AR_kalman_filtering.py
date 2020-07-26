import pandas as pd
from statsmodels.graphics.tsaplots import *
from statsmodels.tsa import arima_model
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import mplfinance as mpf
import talib

def AR_model(trainset):#用于建立AR模型并转为状态空间
    trainset = trainset.values
    AR1 = arima_model.ARIMA(trainset, order=(1, 0, 0)).fit(disp=-1)
    AR2 = arima_model.ARIMA(trainset, order=(2, 0, 0)).fit(disp=-1)
    AR3 = arima_model.ARIMA(trainset, order=(3, 0, 0)).fit(disp=-1)
    AR4 = arima_model.ARIMA(trainset, order=(4, 0, 0)).fit(disp=-1)
    AR5 = arima_model.ARIMA(trainset, order=(5, 0, 0)).fit(disp=-1)
    AR6 = arima_model.ARIMA(trainset, order=(6, 0, 0)).fit(disp=-1)
    AR7 = arima_model.ARIMA(trainset, order=(7, 0, 0)).fit(disp=-1)
    AR8 = arima_model.ARIMA(trainset, order=(8, 0, 0)).fit(disp=-1)
    observation_matrices = np.array([0,0,0,0,0,0,0,1])
    # observation_matrices = AR8.arparams[::-1]
    transition_matrices = np.zeros((8,8))
    transition_matrices[0,0] = AR1.arparams[::-1]
    transition_matrices[1,:2] = AR2.arparams[::-1]
    transition_matrices[2,:3] = AR3.arparams[::-1]
    transition_matrices[3,:4] = AR4.arparams[::-1]
    transition_matrices[4,:5] = AR5.arparams[::-1]
    transition_matrices[5,:6] = AR6.arparams[::-1]
    transition_matrices[6,:7] = AR7.arparams[::-1]
    transition_matrices[7,:8] = AR8.arparams[::-1]
    return observation_matrices,transition_matrices

def kalman_filter(observation_matrices,transition_matrices,trainset,data):#卡尔曼滤波进行过滤
    kf = KalmanFilter(transition_matrices=transition_matrices, observation_matrices=observation_matrices)
    kf.em(trainset)
    filter_mean, filter_cov = kf.filter(trainset)
    index1 = np.where(data.index.month == 4)[0][0]
    for i in range(index1, len(data)):
        next_filter_mean, next_filter_cov = kf.filter_update(
            filtered_state_mean=filter_mean[-1],
            filtered_state_covariance=filter_cov[-1],
            observation=data[i])
        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1, 8, 8)))
    AR_kalman_filter = pd.Series(filter_mean[index1:, 7], index=data.index[index1:])
    return AR_kalman_filter

def kalman_filter_forcast(observation_matrices,transition_matrices,trainset,data):#卡尔曼滤波进行预测
    kf = KalmanFilter(transition_matrices=transition_matrices, observation_matrices=observation_matrices)
    kf.em(trainset)
    filter_mean, filter_cov = kf.filter(trainset)
    index1 = np.where(data.index.month == 4)[0][0]

    for i in range(index1, len(data)):
        next_filter_mean, next_filter_cov = kf.filter_update(
                filtered_state_mean=filter_mean[-1],
                filtered_state_covariance=filter_cov[-1],
                observation=data[i])
        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1, 8, 8)))
    AR_kalman_forcast = np.matmul(np.array(filter_mean) , transition_matrices[7,:])
    # AR_kalman_forcast = pd.Series(AR_kalman_forcast[:-1],index=df.index[1:])
    return AR_kalman_forcast

filename = '600900.SS.csv'
df = pd.read_csv(filename)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index(['Date'], inplace=True)
df['ema12'] = talib.EMA(df.Close,timeperiod=12)
df['ma12'] = talib.MA(df.Close,timeperiod=12)
df['dema12'] = talib.DEMA(df.Close,timeperiod=12)
df['Close_diff1'] = df['Close'].diff(1)

df = df['2019']
# 先一阶差分再过滤
# observation_matrices,transition_matrices = AR_model(df.Close_diff1[:'2019-3'])
# Close_diff1_AR_kalman_filter = kalman_filter(observation_matrices,transition_matrices,
#                                           df.Close_diff1[:'2019-3'],df.Close_diff1)
# Close_AR_kalman_filter = df['Close']['2019-4':] + Close_diff1_AR_kalman_filter

#直接过滤
# observation_matrices,transition_matrices = AR_model(df.Close[:'2019-3'])
# Close_AR_kalman_filter = kalman_filter(observation_matrices,transition_matrices,
#                                            df.Close[:'2019-3'],df.Close)

#预测
observation_matrices,transition_matrices = AR_model(df.Close_diff1[:'2019-3'])
Close_diff1_AR_kalman_forcast = kalman_filter_forcast(observation_matrices,transition_matrices,
                                          df.Close_diff1[:'2019-3'],df.Close_diff1)
Close_diff1_AR_kalman_forcast = pd.Series(Close_diff1_AR_kalman_forcast,index=df.index)
Close_AR_kalman_forcast = df['Close'] + Close_diff1_AR_kalman_forcast

my_color = mpf.make_marketcolors(up='red',
                                 down='green',
                                 edge='i',
                                 volume='in',
                                 inherit=False)

my_style = mpf.make_mpf_style(gridaxis='both',
                              gridstyle=':',
                              facecolor='w',
                              y_on_right=False,
                              marketcolors=my_color)

# add_plot = [mpf.make_addplot(Close_AR_kalman_filter,linestyle='solid',color='y'),
#             mpf.make_addplot(df.ema12['2019-4':],linestyle='solid',color='b')]

add_plot = [mpf.make_addplot(Close_AR_kalman_forcast['2019-4':],linestyle='solid',color='y'),
            mpf.make_addplot(df.ema12['2019-4':],linestyle='solid',color='b')]

mpf.plot(df['2019-4':],type='candle',
         style=my_style,
         addplot=add_plot,
         volume=True,
         figratio=(2,1),
         figscale=5,
)

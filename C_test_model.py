import yfinance as yf
from datetime import date, datetime
from C_indicators import *
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels as sml
import matplotlib.pyplot as plt
import warnings
from itertools import product

#завантажуємо дані
names_list = ['^GSPC', '000001.SS', '^N100', '^HSI', '^KS11', '^N225', '^BSESN']
index_list = []
for i in range(len(names_list)):
	tickerSymbol = names_list[i]
	tickerData = yf.Ticker(tickerSymbol)
	df = tickerData.history(period='1d', start='2018-01-01', end='2020-05-01').fillna(method ='pad')
	df.drop(['Stock Splits', 'Dividends', 'Open'], axis='columns', inplace=True)
	index_list.append(df)

signal_end = []
i = 6
for p in range(len(index_list[i]['Close'])-30):
	s=p
	e=s+30
	data_index = index_list[i]['Close'][s:e]


	index_TA_all_results, list_TA_predict = [], []
	index_TA_all_results.append(strategy_MA_1(data_index, TA_param[i][0], "SMA"))
	index_TA_all_results.append(strategy_MA_1(data_index, TA_param[i][1], "EMA"))
	index_TA_all_results.append(strategy_MA_1(data_index, TA_param[i][2], "LWMA"))
	index_TA_all_results.append(strategy_Aroon(data_index, TA_param[i][3]))
	index_TA_all_results.append(strategy_CCI(data_index, index_list[i]['High'], index_list[i]['Low'], TA_param[i][4]))
	index_TA_all_results.append(strategy_SO(data_index, index_list[i]['High'], index_list[i]['Low'], TA_param[i][5]))
	index_TA_all_results.append(strategy_CMO(data_index, TA_param[i][6]))
	index_TA_all_results.append(strategy_MAE(data_index, upper=0, lower=TA_param[i][7], n1=TA_param[i][8], MA_type="SMA", MAE_type="LL"))
	index_TA_all_results.append(strategy_MAE(data_index, upper=TA_param[i][9], lower=0, n1=TA_param[i][10], MA_type="SMA", MAE_type="UL"))
	index_TA_all_results.append(strategy_MAE(data_index, upper=0, lower=TA_param[i][11], n1=TA_param[i][12], MA_type="LWMA", MAE_type="LL"))
	index_TA_all_results.append(strategy_MAE(data_index, upper=TA_param[i][13], lower=0, n1=TA_param[i][14], MA_type="LWMA", MAE_type="UL"))
	index_TA_all_results.append(strategy_MAE(data_index, upper=0, lower=TA_param[i][15], n1=TA_param[i][16], MA_type="EMA", MAE_type="LL"))
	index_TA_all_results.append(strategy_MAE(data_index, upper=TA_param[i][17], lower=0, n1=TA_param[i][18], MA_type="EMA", MAE_type="UL"))

	#**********
	ARIMA_param = ['n', 'n', 'n', 'n']

	warnings.filterwarnings('ignore')

	p = range(0, 3)
	d = 1
	q = range(0, 3)
	t = ['c', 'nc']
	parameters = product(p, q, t)
	parameters_list = list(parameters)

	best_aic = float("inf")
	for param in parameters_list:
		try:
			model=sml.tsa.arima_model.ARIMA(data_index,	order=(param[0], d, param[1])).fit(disp=-1, trend=param[2])
		except:
			continue
		aic = model.aic
		coeff = True
		
		if aic<best_aic:
			for p in model.pvalues:
				if p > 0.05 and coeff != False:
					coeff = False
		else:
			coeff = False

		if coeff == True:
			best_model = model
			best_aic = model.aic
			ARIMA_param[0] = param[0]
			ARIMA_param[1] = 1
			ARIMA_param[2] = param[1]
			ARIMA_param[3] = param[2]

	if ARIMA_param[0] != 'n':
		forecast = best_model.forecast()
		if forecast[0][0] > data_index[len(data_index)-1]:
			predict_ARIMA = 1
		elif forecast[0][0] < data_index[len(data_index)-1]:
			predict_ARIMA = -1
		else:
			predict_ARIMA = 0
	else:
		predict_ARIMA = 0

	warnings.filterwarnings('default')
	predict_ARIMA_list = []
	predict_ARIMA_list.append(predict_ARIMA)
	index_TA_all_results.append(predict_ARIMA_list)
	#**********

	for j in range(len(index_TA_all_results)):
		list_TA_predict.append(index_TA_all_results[j][len(index_TA_all_results[j])-1])

	for r in range(len(list_TA_predict)):
		if list_TA_predict[r] == 'nan':
			list_TA_predict[r] = 0

	if sum(list_TA_predict) > 1:
		signal_end.append(1)
	elif sum(list_TA_predict) < -1:
		signal_end.append(-1)
	else:
		signal_end.append(0)

#прибираємо повтори
current_element, previous_element = 0, 0
for g in range(len(signal_end)):
	if signal_end[g] != 0:
		previous_element = current_element
		current_element = signal_end[g]
	if previous_element == current_element:
		signal_end[g] = 0

print(test(index_list[i]['Close'][30:], signal_end)-100)
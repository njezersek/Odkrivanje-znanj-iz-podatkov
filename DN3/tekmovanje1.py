import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder

def createLabel(data):
	return data['Route'].astype(str) + "-|-" + data['Route Direction'] + "-|-" + data['Route description'] #+ "-|-" + data['First station'] + "-|-" + data['Last station']

# Reading the data
train = pd.read_csv('train.csv', sep='\t')
test = pd.read_csv('test.csv', sep='\t')

train['label'] = createLabel(train)
test['label'] = createLabel(test)

weather_data = pd.read_csv('weather_data.csv', sep='\t', index_col=0)
prazniki = pd.read_csv('prazniki_prosti.txt', sep='\t', index_col=0, header=None)
pocitnice = pd.read_csv('pocitnice.txt', sep='\t', index_col=0, header=None)


# Converting the data into appropriate format for linear regression

def convert(data):
	d = pd.DataFrame()
	d['departure'] = pd.to_datetime(data['Departure time'])
	d['weekday'] = d['departure'].dt.weekday
	d['month'] = d['departure'].dt.month
	d['hour'] = d['departure'].dt.hour
	d['minute'] = d['departure'].dt.hour*60 + d['departure'].dt.minute
	d['yearday'] = d['departure'].dt.day_of_year
	d['driver'] = data['Driver ID']
	d['weather_key'] = ((d['departure'].copy().round('30min') - pd.to_datetime(date(1800,1,1))).dt.total_seconds()//60).astype(int)
	d['holiday'] = d['departure'].dt.strftime('%Y-%m-%d').isin(prazniki.index).astype(int)
	d['school_holiday'] = d['departure'].dt.strftime('%Y-%m-%d').isin(pocitnice.index).astype(int)

	# merge weather data
	d = pd.merge(d, weather_data, left_on='weather_key', right_index=True, how="left")

	return d

def get_X_train_test(train, test):
	enc = OneHotEncoder(sparse=False)

	d = convert(train.append(test))

	one_hot = enc.fit_transform(d[['weekday', 'hour', 'driver']].values)

	holiday = d[['holiday']].values
	school_holiday = d[['school_holiday']].values
	sunday = np.where(d[['weekday']].values == 6, 1, 0)
	suterday = np.where(d[['weekday']].values == 5, 1, 0)
	weekend = np.where((sunday + suterday) == 1, 1, 0)
	freeday = np.where((holiday + weekend) == 0, 0, 1)
	freezing = np.where(d[['t2m']].values < 0, 1, 0)
	rain = np.where(d[['padavine']].values > 0, 1, 0)
	wind = np.where(d[['veter_hitrost']].values > 1.5, 1, 0)	
	snow = freezing * rain
	t = np.hstack((one_hot, holiday, freeday, school_holiday, freezing, rain, wind, snow))

	return t[:len(train)], t[len(train):], d[:len(train)]

def get_y_train(data):
	d = pd.DataFrame()
	d['departure'] = pd.to_datetime(data['Departure time'])	
	d['arrival'] = pd.to_datetime(data['Arrival time'])
	d['duration'] = (d['arrival'] - d['departure']).dt.total_seconds()

	return d['duration'].to_numpy()


def get_result(data, prediction):
	result = (pd.to_datetime(data['Departure time']) + pd.to_timedelta(prediction, unit='s')).round('1ms')
	return result


def make_prediction(train, test):
	prediction = pd.DataFrame()
	prediction['label'] = test['label']
	prediction['start'] = test['Departure time']
	prediction['duration'] = 0

	routes = {s for s in test['label']} 

	for l in routes:
		X_train, X_test, t = get_X_train_test(train[train['label'] == l], test[test['label'] == l])
		y_train = get_y_train(train[train['label'] == l])
		
		y_pred = np.ones(len(X_test)) * 34 * 60
		if(len(X_train) > 0): # 
			reg = Ridge(alpha=3).fit(X_train, y_train)
			y_pred = reg.predict(X_test)
		else:
			y_pred = np.mean(get_y_train(train)) * np.ones(len(X_test))

		prediction.loc[prediction['label'] == l, 'arrival_prediction'] = get_result(test[test['label'] == l], y_pred)

	prediction['arrival_prediction_round'] = pd.to_datetime(prediction['arrival_prediction']).round('1ms')

	return prediction

# Validate the model with excluding one month from training data

for i in range(1, 12):
	t_test = train[pd.to_datetime(train['Departure time']).dt.month == i]
	t_train = train[pd.to_datetime(train['Departure time']).dt.month != i]

	prediction = make_prediction(t_test, t_test)
	error = (pd.to_datetime(t_test['Arrival time']) - pd.to_datetime(prediction['arrival_prediction_round'])).dt.total_seconds().values

	print(i, np.mean(np.abs(error)))


# Final prediction
prediction = make_prediction(train, test)

prediction[['arrival_prediction_round']].to_csv('result_t_6.txt', index=False, header=False)
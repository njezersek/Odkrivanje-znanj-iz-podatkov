import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Reading the data
train = pd.read_csv('train_pred.csv', sep='\t')
test = pd.read_csv('test_pred.csv', sep='\t')

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
	d['freezing'] = (d['t2m'] < 0).astype(int)
	return d

def get_X_train_test(train, test):
	enc = OneHotEncoder(sparse=False)
	poly_yearday = PolynomialFeatures(degree=2, include_bias=True)
	poly_minute = PolynomialFeatures(degree=4, include_bias=True)
	poly_weather = PolynomialFeatures(degree=2, include_bias=True)
	poly_weekday = PolynomialFeatures(degree=2, include_bias=True)
	scaler_yearday = StandardScaler()
	scaler_minute = StandardScaler()
	scaler_weather = StandardScaler()
	scaler_weekday = StandardScaler()

	d = convert(train.append(test))

	t1 = enc.fit_transform(d[['weekday', 'driver', 'holiday', 'hour', 'freezing', 'school_holiday']].values)

	t2 = d[['yearday']].values
	t2 = scaler_yearday.fit_transform(t2)
	t2 = poly_yearday.fit_transform(t2)
	
	t3 = d[['minute']].values / 60
	t3 = scaler_minute.fit_transform(t3)
	t3 = poly_minute.fit_transform(t3)

	t4 = d[['t2m', 'padavine', 'veter_hitrost']].values
	t4 = scaler_weather.fit_transform(t4)
	t4 = poly_weather.fit_transform(t4)

	t5 = d[['weekday']].values
	t5 = scaler_weekday.fit_transform(t5)
	t5 = poly_weekday.fit_transform(t5)

	t = np.hstack((t1, t2, t3, t4, t5))

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


X_train, X_test, d = get_X_train_test(train, test)
y_train = get_y_train(train)


# Test run

# split into train and test
n = 2000
X_train_eval = X_train[:-n]
y_train_eval = y_train[:-n]
X_train_eval_test = X_train[-n:]
y_train_eval_test = y_train[-n:]

# train the model
reg = Ridge(alpha=3).fit(X_train_eval, y_train_eval)

# predict
y_train_eval_test_pred = reg.predict(X_train_eval_test)

# compute MAE
mae = np.mean(np.abs(y_train_eval_test_pred - y_train_eval_test))

print(f"MAE: {mae}")



# Final prediction on whole date set

reg = Ridge(alpha=3).fit(X_train, y_train)

y_test_pred = reg.predict(X_test)

result = get_result(test, y_test_pred)
result.to_csv('result7.txt', index=False, header=False)
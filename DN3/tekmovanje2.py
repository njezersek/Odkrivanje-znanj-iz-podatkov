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

train = pd.read_csv('train.csv', sep='\t')
test = pd.read_csv('test.csv', sep='\t')

train['label'] = train['First station'] + "---" + train['Last station']
test['label'] = test['First station'] + "---" + test['Last station']

weather_data = pd.read_csv('weather_data.csv', sep='\t', index_col=0)
prazniki = pd.read_csv('prazniki_prosti.txt', sep='\t', index_col=0, header=None)
pocitnice = pd.read_csv('pocitnice.txt', sep='\t', index_col=0, header=None)

train_routes = {s for s in train['label']}
test_routes = {s for s in test['label']}

# Compute mean and median for each line
routes_avg = pd.DataFrame({"label": list(test_routes | train_routes)})
routes_avg['duration'] = 0
routes_avg.set_index('label', inplace=True)

d = pd.DataFrame()
d['label'] = train['label']
d['departure'] = pd.to_datetime(train['Departure time'])	
d['arrival'] = pd.to_datetime(train['Arrival time'])
d['duration'] = (d['arrival'] - d['departure']).dt.total_seconds()

for l in routes_avg.index:
	routes_avg.loc[l, 'med_duration'] = np.median(d.loc[d['label'] == l, 'duration'])
	routes_avg.loc[l, 'avg_duration'] = np.mean(d.loc[d['label'] == l, 'duration'])

routes_avg.loc['Tbilisijska---VIŽMARJE', 'avg_duration'] = 34*60


# Mearge mean and median with the data set
train = pd.merge(train, routes_avg, left_on='label', right_index=True, how="left")
test = pd.merge(test, routes_avg, left_on='label', right_index=True, how="left")

# Convert the data to appropriate format for linear regression

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
	d['label'] = data['First station'] + "---" + data['Last station']

	# merge weather data
	d = pd.merge(d, weather_data, left_on='weather_key', right_index=True, how="left")
	# d = pd.merge(d, routes_avg, left_on='label', right_index=True, how="left")
	d['freezing'] = (d['t2m'] < 0).astype(int)
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
	d['duration_delta'] = d['duration'] - data['med_duration']
	d['delta_k'] = d['duration'] / data['med_duration']
	return d['delta_k'].to_numpy()


def get_result(data, prediction):
	result = (pd.to_datetime(data['Departure time']) + pd.to_timedelta(prediction, unit='s')).round('1ms')
	return result


X_train, X_test, d = get_X_train_test(train, test)
y_train = get_y_train(train)

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
duration_pred = y_train_eval_test_pred * train[-n:]['avg_duration'].values
duration_test = y_train_eval_test * train[-n:]['avg_duration'].values
mae = np.mean(np.abs(duration_pred - duration_test))

print(f"MAE: {mae}")

# Final prediction
reg = Ridge(alpha=3).fit(X_train, y_train)

y_pred = reg.predict(X_test)

duration_pred = y_pred * test['avg_duration'].values

prediction = get_result(test, duration_pred)

prediction.to_csv('result_t_2_2.txt', index=False, header=False)


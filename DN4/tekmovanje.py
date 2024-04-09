import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

def createLabel(data):
	return data['Route'].astype(str) + "-|-" + data['Route Direction'] + "-|-" + data['Route description'] #+ "-|-" + data['First station'] + "-|-" + data['Last station']

# read the data
train = pd.read_csv('train.csv', sep='\t')
test = pd.read_csv('test.csv', sep='\t')

train['label'] = createLabel(train)
test['label'] = createLabel(test)

weather_data = pd.read_csv('weather_data.csv', sep='\t', index_col=0)
prazniki = pd.read_csv('prazniki_prosti.txt', sep='\t', index_col=0, header=None)
pocitnice = pd.read_csv('pocitnice.txt', sep='\t', index_col=0, header=None)

train_routes = {s for s in train['label']}
test_routes = {s for s in test['label']}


# preprocessing

def convert(data):
	d = pd.DataFrame()
	d['departure'] = pd.to_datetime(data['Departure time'])
	d['weekday'] = d['departure'].dt.weekday
	d['month'] = d['departure'].dt.month
	d['hour'] = d['departure'].dt.hour
	d['minute_of_day'] = d['departure'].dt.hour*60 + d['departure'].dt.minute
	d['second_of_day'] = d['departure'].dt.hour*3600 + d['departure'].dt.minute*60 + d['departure'].dt.second
	d['day_of_year'] = d['departure'].dt.dayofyear
	d['driver'] = data['Driver ID']
	d['driver_l'] = np.where(d['driver'] < 254, d['driver'], 254)
	d['driver_h'] = np.where(d['driver'] >= 255, d['driver']-255, 254)
	d['weather_key'] = ((d['departure'].copy().round('30min') - pd.to_datetime(date(1800,1,1))).dt.total_seconds()//60).astype(int)
	d['holiday'] = d['departure'].dt.strftime('%Y-%m-%d').isin(prazniki.index).astype(int)
	d['school_holiday'] = d['departure'].dt.strftime('%Y-%m-%d').isin(pocitnice.index).astype(int)
	d['label'] = data['First station'] + "---" + data['Last station']

	# seconds on the circle
	sec = (d["second_of_day"] / (24*60*60)) * 2 * np.pi
	d["sec_circ_x"] = np.cos(sec)
	d["sec_circ_y"] = np.sin(sec)

	# month on the circle
	month = (d["month"] / 12) * 2 * np.pi
	d["month_circ_x"] = np.cos(month)
	d["month_circ_y"] = np.sin(month)

	# day of year on the circle
	day = (d["day_of_year"] / 365) * 2 * np.pi
	d["day_circ_x"] = np.cos(day)
	d["day_circ_y"] = np.sin(day)

	# merge weather data
	d = pd.merge(d, weather_data, left_on='weather_key', right_index=True, how="left")
	d['freezing'] = (d['t2m'] < 0).astype(int)
	return d

def get_X_train_test(train, test):
	d = convert(train.append(test))
	t = d[["driver", "weekday", "month", "freezing", "school_holiday", "holiday", "sec_circ_x", "sec_circ_y", "day_circ_x", "day_circ_y", "padavine"]].values
	# t = d[["weekday", "month", "freezing", "school_holiday", "holiday", "second_of_day", "day_of_year", "t2m", "padavine", "veter_hitrost"]].values

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
			reg = RandomForestRegressor().fit(X_train, y_train)
			y_pred = reg.predict(X_test)
		else:
			y_pred = np.mean(get_y_train(train)) * np.ones(len(X_test))

		prediction.loc[prediction['label'] == l, 'arrival_prediction'] = get_result(test[test['label'] == l], y_pred)

	prediction['arrival_prediction_round'] = pd.to_datetime(prediction['arrival_prediction']).round('1ms')

	return prediction



# make prediction

prediction = make_prediction(train, test)
prediction[['arrival_prediction_round']].to_csv('result_t3_7.txt', index=False, header=False)
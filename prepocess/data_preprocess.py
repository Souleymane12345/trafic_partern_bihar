import pandas as pd
import sqlite3
import numpy as np
import common

class step_0:

    def pickup_datetime(data):
        data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
        return data


    def load_data(path, data_type):
        print(f"Reading train data from the database: {path}")
        con = sqlite3.connect(path)
        data_type = pd.read_sql('SELECT * FROM train', con)
        con.close()
        X = data_type.drop(columns=['target'])
        y = data_type['target']
        return X, y

    def transform_target(y):
        y = np.log1p(y).rename('log_'+y.name)
        return y


class step_adding_features:

# --------------Step 1------------- 

    def step1_add_features(X):
        
        df_abnormal_dates = X.groupby('pickup_date').size()
        abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]
        res = X.copy()
        
        res['weekday'] = res['pickup_datetime'].dt.weekday
        res['month'] = res['pickup_datetime'].dt.month
        res['hour'] = res['pickup_datetime'].dt.hour
        res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
        return res

# --------------Step 2------------- 

    def haversine_array(self, lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def is_high_traffic_trip(self, X):
        return ((X['hour'] >= 8) & (X['hour'] <= 19) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
                ((X['hour'] >= 13) & (X['hour'] <= 20) & (X['weekday'] == 5))

    def is_high_speed_trip(self, X):
        return ((X['hour'] >= 2) & (X['hour'] <= 5) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
                ((X['hour'] >= 4) & (X['hour'] <= 7) & (X['weekday'] >= 5) & (X['weekday'] <= 6))
                
    
    def is_rare_point(self, X, latitude_column, longitude_column, qmin_lat, qmax_lat, qmin_lon, qmax_lon):
        lat_min = X[latitude_column].quantile(qmin_lat)
        lat_max = X[latitude_column].quantile(qmax_lat)
        lon_min = X[longitude_column].quantile(qmin_lon)
        lon_max = X[longitude_column].quantile(qmax_lon)

        res = (X[latitude_column] < lat_min) | (X[latitude_column] > lat_max) | \
                (X[longitude_column] < lon_min) | (X[longitude_column] > lon_max)
        return res

    def step2_add_features(self, X):
        res = X.copy()
        distance_haversine = self.haversine_array(res.pickup_latitude, res.pickup_longitude, res.dropoff_latitude, res.dropoff_longitude)
        res['log_distance_haversine'] = np.log1p(distance_haversine)
        res['is_high_traffic_trip'] = self.is_high_traffic_trip(X).astype(int)
        res['is_high_speed_trip'] = self.is_high_traffic_trip(X).astype(int)
        res['is_rare_pickup_point'] = self.is_rare_point(X, "pickup_latitude", "pickup_longitude", 0.01, 0.995, 0, 0.95).astype(int)
        res['is_rare_dropoff_point'] = self.is_rare_point(X, "dropoff_latitude", "dropoff_longitude", 0.01, 0.995, 0.005, 0.95).astype(int)

        return res

# --------------Step 3------------- 

    def step3_process_features(X):
        res = X.copy()
        res['vendor_id'] = res['vendor_id'].map({1:0, 2:1})
        res['store_and_fwd_flag'] = res['store_and_fwd_flag'].map({'N':0, 'Y':1})
        res.loc[res['passenger_count'] > 6, 'passenger_count'] = 6

        return res


# --------------Step 4------------- 

    class step4_adding_features:

        def step4_remove_outliers(X, y):
            res = X.copy()

            d = res['log_distance_haversine']
            d_min = d.quantile(0.01)
            d_max = d.quantile(0.995)
            cond_distance = (d > d_min) & (d < d_max)

            y_min = y.quantile(0.005)
            y_max = y.quantile(0.995)
            cond_time = (y > y_min) & (y < y_max)

            cond_non_outliers = cond_distance & cond_time

            return X[cond_non_outliers], y[cond_non_outliers]


# --------------Step 5------------- 

    def encode_weekday(self, x):
        if (x == 0): return 0
        if (x == 5): return 2
        if (x == 6): return 3
        return 1

    def step5_process_categorical_features(self, X):
        res = X.copy()
        res['weekday'] = res['weekday'].apply(self.encode_weekday)

        return res








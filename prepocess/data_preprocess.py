import pandas as pd



def pickup_datetime(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    return data
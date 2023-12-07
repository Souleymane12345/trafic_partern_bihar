import requests
import zipfile
import io
import common
import sqlite3
import os
import pandas as pd


from prepocess.data_preprocess import step_0
from sklearn.model_selection import train_test_split



url = common.DATA_PATH
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    #print(common.ZIP_PATH)
    z.extractall(common.ZIP_PATH)

data = pd.read_csv(common.DATA_FILE_PATH)
data = step_0.pickup_datetime(data)

X = data.drop(columns=['trip_duration'])

y = data['trip_duration']
data_train, data_test = train_test_split(data, test_size=0.3, random_state=common.RANDOM_STATE)



db_dir = os.path.dirname(common.DB_PATH)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

print(f"Saving train and test data to a database: {common.DB_PATH}")
with sqlite3.connect(common.DB_PATH) as con:

    data_train.to_sql(name='train', con=con, if_exists="replace")
    data_test.to_sql(name='test', con=con, if_exists="replace")
    
print(f"Reading train data from the database: {common.DB_PATH}")
with sqlite3.connect(common.DB_PATH) as con:
    cur = con.cursor()
    res = cur.execute("SELECT * FROM train LIMIT 3")
    print(res.fetchall())
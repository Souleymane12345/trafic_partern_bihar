import pickle
import os

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config/config.ini')

# Using INI configuration file
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)

#print(ROOT_DIR)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
#MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))

RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))

DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))

TARGET = 'trip_duration'

STEP_0_NUMFEATURES  = ['abnormal_period', 'hour']
STEP_0_CAT_FEATURES  = ['weekday', 'month']

DATA_PATH = "https://github.com/eishkina-estia/ML2023/raw/main/data/New_York_City_Taxi_Trip_Duration.zip"

DATA_FILE_PATH = os.path.join(ROOT_DIR, 'connection/data/New_York_City_Taxi_Trip_Duration.csv')

ZIP_PATH = os.path.join(ROOT_DIR, 'connection/data/')

MODEL_CREATE_PATH = os.path.join(ROOT_DIR, 'model/')

TRAIN = 'train'
TEST = 'test'
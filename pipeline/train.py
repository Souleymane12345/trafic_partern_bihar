from prepocess.data_preprocess import *
from utils.utils_t import utils_tools
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


X_train, y_train = step_0.load_data(common.DB_PATH, common.TRAIN )
X_train = step_0.pickup_datetime(X_train)
y_train = step_0.transform_target(y_train)


num_features = common.STEP_0_NUMFEATURES

cat_features = common.STEP_0_CAT_FEATURES

train_features = num_features + cat_features

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
)

pipeline = Pipeline(steps=[
    ('ohe_and_scaling', column_transformer),
    ('regression', Ridge())
])

X_train = step_adding_features.step1_add_features(X_train)
y_train = step_0.transform_target(y_train)


model = pipeline.fit(X_train[train_features], y_train)
y_pred_train = model.predict(X_train[train_features])

model_path = common.MODEL_CREATE_PATH + 'model.pkl'
utils_tools.persist_model(model,model_path)

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))

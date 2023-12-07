from prepocess.data_preprocess import *
from utils.utils_t import utils_tools
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


X_test, y_test = step_0.load_data(common.DB_PATH, common.TEST)
X_test = step_0.pickup_datetime(X_test)
y_test = step_0.transform_target(y_test)


X_test = step_adding_features.step1_add_features(X_test)


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

model_path = common.MODEL_CREATE_PATH + 'model.pkl'

model = utils_tools.load_model(model_path)


y_pred_test = model.predict(X_test[train_features])
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))
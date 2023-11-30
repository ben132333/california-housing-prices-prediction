import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

"""
Load and prepare the train/test data
"""

# Read the data from the csv file
housing_districts_total = pd.read_csv('housing.csv')

train_set, test_set = train_test_split(
    housing_districts_total, 
    test_size=0.2, 
    random_state=23
)
print(len(train_set), "train |", len(test_set), "test")

housing_districts_training = train_set.copy()
housing_districts_test = test_set.copy()

""" 
The data processing pipeline
"""

def add_features(df):
    # Larger homes tend to be more expensive so rooms per household
    # might be a useful attribute. 
    df["rooms_per_household"] = \
    df["total_rooms"]/df["households"]

    # Bedrooms per household might also be a useful attribute.
    df["bedrooms_per_household"] = \
    df["total_bedrooms"]/df["households"]

    # Bigger households might mean more expensive houses.
    df["population_per_household"] = \
    df["population"]/df["households"]

    return df

# Pipeline for cleaning, feature engineering and preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

cat_pipeline = make_pipeline(
    OneHotEncoder(handle_unknown="ignore")
)

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income",
               "rooms_per_household", "bedrooms_per_household", "population_per_household"]
cat_attribs = ["ocean_proximity"]

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)],
    verbose_feature_names_out=False
)

housing_districts = housing_districts_training.drop("median_house_value", axis=1)
housing_districts_labels = housing_districts_training["median_house_value"].copy()

housing_districts = add_features(housing_districts)
housing_districts_prepared = preprocessing.fit_transform(housing_districts)

# Turn the prepared data into a dataframe:
# housing_districts_prepared_df = pd.DataFrame(
#     data=housing_districts_prepared,
#     columns=preprocessing.get_feature_names_out(),
#     index=housing_districts.index
# )

"""
Training the model
"""

# Train model:
rfr = RandomForestRegressor(n_estimators=100, random_state=23)
rfr.fit(housing_districts_prepared, housing_districts_labels)

# Save the trained model:
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(rfr, f"rfr_{current_datetime}.pkl")

# Load model: 
# rfr = joblib.load("./rfr1.pkl")

# Test loaded model on example:
# example = housing_districts_prepared_df.values[1]
# print(rfr.predict([example]))

"""
Model evaluation
"""

housing_districts_predictions = rfr.predict(housing_districts_prepared)
rfr_mse = mean_squared_error(housing_districts_labels, housing_districts_predictions)
rfr_rmse = np.sqrt(rfr_mse)

print("RMSE:", rfr_rmse)

# use cross validation score if you're not ready to use the test set:
# scores = cross_val_score(
#     rfr, 
#     housing_districts_prepared, 
#     housing_districts_labels,
#     scoring="neg_mean_squared_error", 
#     cv=5
# )
# rfr_rmse_scores = np.sqrt(-scores)
# print("Mean RMSE with 10-fold cross validation:", rfr_rmse_scores.mean())

# Test the model on the test set:
housing_districts_test = housing_districts_test.drop("median_house_value", axis=1)
housing_districts_test_labels = test_set["median_house_value"].copy()

housing_districts_test = add_features(housing_districts_test)
housing_districts_test_prepared = preprocessing.transform(housing_districts_test)

housing_districts_test_predictions = rfr.predict(housing_districts_test_prepared)
rfr_test_mse = mean_squared_error(housing_districts_test_labels, housing_districts_test_predictions)
rfr_test_rmse = np.sqrt(rfr_test_mse)

print("RMSE on the test set:", rfr_test_rmse)

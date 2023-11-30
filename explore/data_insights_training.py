import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def clean_date(df):
    # Handle missing values for the total_bedrooms attribute
    total_bedroom_median = df["total_bedrooms"].median()
    df["total_bedrooms"].fillna(total_bedroom_median, inplace=True)
    return total_bedroom_median, df

# Read the data from the csv file
housing_districts_total = pd.read_csv('housing.csv')

train_set, test_set = train_test_split(
    housing_districts_total, 
    test_size=0.2, 
    random_state=23
)
print(len(train_set), "train |", len(test_set), "test")

housing_districts_training = train_set.copy()

housing_districts = housing_districts_training.drop("median_house_value", axis=1)
housing_districts_labels = housing_districts_training["median_house_value"].copy()

# Look for correlations

# Let's check the linear correlation coefficient between 
# each pair of numerical attributes.
housing_districts_numerical = \
housing_districts.drop("ocean_proximity", axis=1)

corr_matrix = housing_districts_numerical.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing_districts.plot(
    kind="scatter", 
    x="median_income", 
    y="median_house_value"
    # alpha=0.1
)

plt.show()

total_bedroom_median, housing_districts = clean_date(housing_districts)

# Histogram of the total number of rooms per household
print(housing_districts.info())
print("Median:", (total_bedroom_median))

# Feature engineering

# Larger homes tend to be more expensive so rooms per household
# might be a useful attribute. 
housing_districts["rooms_per_household"] = \
housing_districts["total_rooms"]/housing_districts["households"]

# Bedrooms per household might also be a useful attribute.
housing_districts["bedrooms_per_household"] = \
housing_districts["total_bedrooms"]/housing_districts["households"]

Handle categorical attributes
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

tr_ocean_proximity = encoder.fit_transform(housing_districts[["ocean_proximity"]])
df_ocean_proximity = pd.DataFrame(tr_ocean_proximity, 
                                  columns=encoder.get_feature_names_out(),
                                  index=housing_districts.index)

housing_districts = pd.concat([housing_districts, df_ocean_proximity], axis=1)
housing_districts.drop("ocean_proximity", axis=1, inplace=True)

print(housing_districts.head())
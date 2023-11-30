import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the csv file
housing_districts = pd.read_csv('housing.csv')

# Explore the dataset
print("The first 5 rows of the dataset:")
print(housing_districts.head())

print("\nDataframe info:")
print(housing_districts.info())

# ocean_proximity is the only non-numerical attribute. 
# Let's explore it.
print("\nValue counts of the ocean_proximity attribute:")
print(housing_districts['ocean_proximity'].value_counts())

print("\nSummary of the numerical attributes:")
print(housing_districts.describe())

housing_districts_copy = housing_districts.copy()

# Scatter plot of the median house price of each district. The x axis is the longitude. The y axis is the latitude. The size of the circle represents the population of the district. The color represents the median house price.
housing_districts_copy.plot(
    kind="scatter", 
    x="longitude", y="latitude", grid=True,
    s=housing_districts_copy["population"]/200, label="population",
    c="median_house_value", cmap="jet", colorbar=True,
    legend=True, sharex=False, figsize=(10, 7)
)
# plt.savefig('scatter_plot_prices.png')

# Histograms to explore the data
# housing_districts_copy.hist(
#     bins=50, 
#     figsize=(15, 7)
# )

plt.show()

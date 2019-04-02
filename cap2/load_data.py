import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix


HOUSING_PATH = "datasets/housing"

def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)

def plot_hist(housing):
	housing.hist(bins=50, figsize=(20,15))
	plt.savefig("attribute_histogram_plots")
	plt.show()

def split_data_strat(housing):
	housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(housing, housing["income_cat"]):
	    strat_train_set = housing.loc[train_index]
	    strat_test_set = housing.loc[test_index]

	#print("test set")
	#print(strat_test_set["income_cat"].value_counts()/ len(strat_test_set) )
	#print("all dataset")
	#print(housing["income_cat"].value_counts()/ len(housing))

	return strat_train_set,strat_test_set

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

def compare_data(housing,strat_test_set):
	train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

	compare_props = pd.DataFrame({
	    "Overall": income_cat_proportions(housing),
	    "Stratified": income_cat_proportions(strat_test_set),
	    "Random": income_cat_proportions(test_set),
	}).sort_index()
	compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
	compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

	print(compare_props)

def delete_attribute(name):
	for set_ in (strat_train_set, strat_test_set):
		set_.drop(name, axis=1, inplace=True)
	return strat_train_set, strat_train_set

def plot_lat_long(housing):
	housing.plot(kind="scatter", x="longitude", y="latitude")
	plt.show()
	plt.savefig("bad_visualization_plot")

	housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.5)
	plt.show()
	plt.savefig("better_visualization_plot")

def plot_prices_scatterplot(housing):
	housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
	plt.legend()
	plt.savefig("housing_prices_scatterplot")
	plt.show()

def plot_scatter_matrix(housing):
	attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
	scatter_matrix(housing[attributes], figsize=(12, 8), color="black")
	plt.savefig("scatter_matrix_plot")
	plt.show()


housing = load_housing_data()
#print(housing.head())
#print(housing.info())
#print(housing["ocean_proximity"].value_counts())
#print(housing.describe())

#plot_hist(housing)

#housing.hist(bins=50, figsize=(20,15))
#plt.savefig("attribute_histogram_plots")
#plt.show()

#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print(test_set.head())



#print(housing["income_cat"].value_counts())

#print(housing.head())
#print(housing.info())

#housing["income_cat"].hist()
#plt.show()

strat_train_set, strat_test_set = split_data_strat(housing)

#compare_data(housing,strat_test_set)

strat_train_set, strat_train_set = delete_attribute("income_cat")

housing = strat_train_set.copy()

#plot_lat_long(housing)
#plot_prices_scatterplot(housing)

#corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#plot_scatter_matrix(housing)

#housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.axis([0, 16, 0, 550000])
#plt.savefig("income_vs_house_value_scatterplot")
#plt.show()

#housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
#housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
#housing["population_per_household"]=housing["population"]/housing["households"]

#corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))









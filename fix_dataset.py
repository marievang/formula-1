import os
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Replace 'your_file.tsv' with the path to your tab-separated file
file_path = '/home/emikot/Desktop/linux/Python/chav/final_project/maria.tsv'

# Read the tab-separated file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Display the first few rows of the DataFrame
print(df.head())

#import csv
os.chdir("/home/emikot/Desktop/linux/Python/chav/final_project/Downloaded_data")

#get drivers
my_drivers = pd.read_csv("my_drivers.csv")
print(my_drivers)


#MERGE constructorId, constructorRef
df = df.merge(my_drivers, on="code", how="left")

#DRIVERS ARE NOT UNIQUE

#Race results
race_results = pd.read_csv("results.csv")
race_results = race_results[['driverId', 'constructorId']]
f_race_re = race_results.tail(20)
#race_results.rename(columns={'position': 'race_position'}, inplace=True)

df = df.merge(f_race_re, on="driverId", how="left")

#get constructors
all_constructors = pd.read_csv("constructors.csv")
all_constructors = all_constructors[["constructorId","constructorRef"]]
print(all_constructors)

df = df.merge(all_constructors, on="constructorId", how="left")

#Read the circuit data
circuits = pd.read_csv("circuits.csv")
circuits = circuits[['circuitId', 'country']]

df = df.merge(circuits, on="circuitId", how="left")

df= df.fillna(0)

#Convert q3 from minutes to seconds. Q3 example: 1:20.142
def convert_q3_to_seconds(q3):
	if pd.isna(q3) or q3 == 0:
		return 0
	else:
		minutes, seconds = q3.split(":")
		return int(minutes) * 60 + float(seconds)

df["q3"] = df["q3"].apply(convert_q3_to_seconds)

df.loc[:, "home_advantage"] = df.apply(lambda x: 1 if x["nationality"] == x["country"] else 0, axis=1)


def merge_team_ids(x):
	if x in [10, 211]:
		return 117
	elif x == 15:
		return 51
	elif x in [4,208]:
		return  214
	elif x in [5,213]:
		return 215
	else:
		return x

df["constructorId"] = df["constructorId"] .apply(merge_team_ids)

#Read the driver and constructor error rates from the csv
driver_error_rates = pd.read_csv("driver_error_rates.csv", usecols=lambda column: column not in ["Unnamed: 0"])
constructor_error_rates = pd.read_csv("constructor_error_rates.csv", usecols=lambda column: column not in ["Unnamed: 0"])

df = pd.merge(df, driver_error_rates, on=["code"])
df = pd.merge(df, constructor_error_rates, on=["constructorId"])

results = df[["raceId","code","driverId","race_position"]]

df.drop(["code", "constructorRef", "nationality", "country", "DNF"], axis=1, inplace=True)


#Change the column order
df = df[['raceId', 'driverId', 'constructorId', 'grid', 'standings_points', 'standings_position', 'standings_wins', 'points', 'constructor_position', 'wins', 'year', 'circuitId', 'qualifying_position', 'q3', 'DriverErrorRate', 'ConstrErrorRate', 'home_advantage']]

#rename
df.rename(columns={'points': 'constructor_points', 'wins': 'constructor_wins'}, inplace=True)

# Save the DataFrame to a csv file
df.to_csv("fixed_dataset.csv", index=False)


#copy the results to another dataframe
podium = results.copy() 
winner = results.copy()

podium["race_position"]	= podium["race_position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

podium.sort_values("raceId", inplace=True)
podium.to_csv("podium.csv", index=False)


winner["race_position"] = winner["race_position"].apply(lambda x: 1 if x ==1 else 0)


winner.sort_values("raceId", inplace=True)

winner.to_csv("winner.csv", index=False)

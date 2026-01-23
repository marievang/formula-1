###########################################################
#import race data from csv
import os
import pandas as pd

#Nottes
#Files downloaded and used: results.csv, drivers.csv, constructors.csv, driver_standings.csv, constructor_standings.csv, races.csv, circuits.csv, qualifying.csv 
#FIles created and used: driver_error_rates.csv, constructor_error_rates.csv" -- only for the current drivers
#Merged same constructors
#q3 changed from minutes to seconds

#pd.set_option('display.max_rows', 50)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

#Set directory
path_to_files = "/home/emikot/Desktop/linux/Python/chav/final_project/Downloaded_data"
os.chdir(path_to_files)

#Import available data

#Race results
race_results = pd.read_csv("results.csv")
race_results = race_results[['raceId', 'driverId', 'constructorId','resultId','grid','position']]
race_results.rename(columns={'position': 'race_position'}, inplace=True)
print(race_results)

#Driver data
all_drivers = pd.read_csv("drivers.csv")
all_drivers = all_drivers[["driverId","code","nationality"]]
print(all_drivers)

#Constructors data
all_constructors = pd.read_csv("constructors.csv")
all_constructors = all_constructors[["constructorId","constructorRef"]]
print(all_constructors)

#Driver data
driver_standings = pd.read_csv("driver_standings.csv")
driver_standings = driver_standings[['raceId', 'driverId', 'points', 'position','positionText', 'wins']]
driver_standings.rename(columns={'wins': 'standings_wins', 'points': 'standings_points', 'position': 'standings_position'}, inplace=True)
print(driver_standings)

#Constructors satndings
constructor_standings = pd.read_csv("constructor_standings.csv")
constructor_standings = constructor_standings[['raceId', 'constructorId', 'points', 'position','wins']]
constructor_standings.rename(columns={'wins': 'constructor_wins', 'points': 'constructor_points', 'position': 'constructor_position'}, inplace=True)

#Races
races = pd.read_csv("races.csv")
races = races[['raceId', 'year','circuitId']]

#Circuit data
circuits = pd.read_csv("circuits.csv")
circuits = circuits[['circuitId', 'country']]

#Qualifying results
quali_results = pd.read_csv("qualifying.csv")	
quali_results = quali_results[['raceId', 'driverId', 'constructorId', 'number' ,'position','q3']]
quali_results.rename(columns={'position': 'qualifying_position'}, inplace=True)

#Read the driver and constructor error rates from the csv files
driver_error_rates = pd.read_csv("driver_error_rates.csv", usecols=lambda column: column not in ["Unnamed: 0"])
constructor_error_rates = pd.read_csv("constructor_error_rates.csv", usecols=lambda column: column not in ["Unnamed: 0"])

#Function for merging constructor ids
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

#Merge all available data to a dataframe
data1 = pd.merge(race_results, all_drivers, on=["driverId"])
data2 = pd.merge(data1, all_constructors, on=["constructorId"])
data3 = pd.merge(data2, driver_standings, on=["raceId", "driverId"])
data4 = pd.merge(data3, constructor_standings, on=["raceId", "constructorId"])
data5 = pd.merge(data4, races, on=["raceId"])
dataset = pd.merge(data5, quali_results, on=["raceId", "driverId","constructorId"])
#merge the constructor ids
dataset["constructorId"] = dataset["constructorId"] .apply(merge_team_ids)

#Create a csv with all races and drivers from 1950
dataset.to_csv("all_dataset.csv", sep=',', index=False, encoding='utf-8')

#Add error rates
dataset1 = pd.merge(dataset, driver_error_rates, on=["code"])
dataset2 = pd.merge(dataset1, constructor_error_rates, on=["constructorId"])
dataset3 = pd.merge(dataset2, circuits, on=["circuitId"])

#Do not need this theerror rates are calculated only for the current drivers
#current_drivers = ["VER","NOR","LEC","PIA","SAI","HAM","PER","RUS","ALO","STR","HUL","TSU","RIC","GAS","BEA","MAG","ALB","OCO","ZHO","SAR","BOT"]
#keep only the dataframe rows with the current drivers in the code column
#dataset_curr_dr = dataset[dataset['code'].isin(current_drivers)]
#dataset_curr_dr.to_csv("2018_curr_dr_dataset.csv", sep=',', index=False, encoding='utf-8')

dataset_curr_dr = dataset3
dataset_curr_dr.replace(r'\N', 0, inplace=True)

#Convert q3 from minutes to seconds. Q3 example: 1:20.142
def convert_q3_to_seconds(q3):
	if pd.isna(q3) or q3 == 0:
		return 0
	else:
		minutes, seconds = q3.split(":")
		return int(minutes) * 60 + float(seconds)

dataset_curr_dr["q3"] = dataset_curr_dr["q3"].apply(convert_q3_to_seconds)

#REMOVE THE POINTS -- DONE
#ADD DRIVER AND COSTRUCTOR ERROR -- DONE
#NEED TO CHANGE TIME TO FLOAT -- DONE
#NATIONALITY - HOME ADVVANTAGE -- DONE


#CREATE PREDICTION COLUMN

#dataset_curr_dr.loc[:, "y"] = dataset_curr_dr["race_position"].apply(lambda x: 1 if x == "1" else 0)

#Replace all "/N" with 0


driver_nationalities = dataset_curr_dr["nationality"].unique()

#Make a new column called home advantage that is 1 if the driver is racing in their home country and 0 otherwise
dataset_curr_dr.loc[:, "home_advantage"] = dataset_curr_dr.apply(lambda x: 1 if x["nationality"] == x["country"] else 0, axis=1)

dataset_curr_dr=dataset_curr_dr.drop(["nationality","country","positionText","resultId"], axis=1)
#Create a csv file with the final dataset
dataset_curr_dr.to_csv("current_driver_dataset.csv", sep=',', index=False, encoding='utf-8')


#import race data from csv
import os
import pandas as pd
import matplotlib.pyplot as plt

#import csv
os.chdir("/home/artemis/Desktop/linux/Python/chav/final_project/Downloaded_data")

#Race results
race_results = pd.read_csv("results.csv")
race_results = race_results[['raceId', 'driverId', 'constructorId','statusId']]
print(race_results)

#get drivers
all_drivers = pd.read_csv("drivers.csv")
all_drivers = all_drivers[["driverId","number","code"]]
print(all_drivers)

#get constructors
all_constructors = pd.read_csv("constructors.csv")
all_constructors = all_constructors[["constructorId","constructorRef"]]
print(all_constructors)

#Finishing status
status = pd.read_csv("status.csv")
print(status)
#Driver error: 3,4,20,104,130,137,139,100,82
#Costructor error: everything but: 1,2,3,4,20,104,130,137,139,100,11,12,13,14,15,16,17,18,19,27,29,45,50,128,53,55,58,62,81,82,88,89,90,97,111,112,113,115,116,117,118,119,120,121,122,123,124,125,127,133,134.

driver_err_ids = [3,4,20,104,130,137,139,100,82]
constr_no_err_ids = [1,2,3,4,20,104,130,137,139,100,11,12,13,14,15,16,17,18,19,27,29,45,50,128,53,55,58,62,81,82,88,89,90,97,111,112,113,115,116,117,118,119,120,121,122,123,124,125,127,133,134]

just_driverIds = all_drivers["driverId"]
driver_codes = all_drivers["code"]

#Keep only the current driver's error rates

current_drivers = ["VER","NOR","LEC","PIA","SAI","HAM","PER","RUS","ALO","STR","HUL","TSU","RIC","GAS","BEA","MAG","ALB","OCO","ZHO","SAR","BOT"]
current_driver_rates = []


driver_rates= []
driver_statuses ={}
all_driver_errors = pd.Series(dtype=int)

for driverId in just_driverIds:
	curr_driver_rr = race_results.loc[race_results['driverId'] == driverId]
	total_curr_dr_races = len(curr_driver_rr)
	curr_driver_errors_rr = curr_driver_rr.loc[curr_driver_rr['statusId'].isin(driver_err_ids)]
	
	curr_driver_error_rate = len(curr_driver_errors_rr)/total_curr_dr_races
	
	curr_driver_errors = curr_driver_errors_rr.statusId.value_counts()
	all_driver_errors = all_driver_errors.add(curr_driver_errors, fill_value=0)

	driver_rates.append(curr_driver_error_rate)


print(driver_rates)
################################################################3
# Calculate the percentage of each error
error_percentage = all_driver_errors / all_driver_errors.sum() * 100


#####################################################################
code_list = list(driver_codes)
ids_list = list(just_driverIds)

for a_curr_driver in current_drivers:
	dr_index = code_list.index(a_curr_driver)
	#curr_dr_id = ids_list[dr_index]
	current_driver_rates.append(driver_rates[dr_index])
	


print(current_driver_rates)


#Plot the error rates of all drivers


fig, ax = plt.subplots()
ax.bar(current_drivers, current_driver_rates)
ax.set_xlabel('Driver ID')
ax.set_ylabel('Error Rate')
ax.set_title('Error Rate of all drivers')

plt.show()

#MEGRE CONSTRUCTORS
#merge alpha--sauber, - rb, torro, racing point--force india--aston martin, alpine--renault--lotus f1, rb--torro rosso--alphatauri
# Toleman, Benetton Formula, Lotus F1, Renault in Formula One

# Minardi, Scuderia Toro Rosso, and Scuderia AlphaTauri

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
	

race_results["constructorId"] = race_results["constructorId"] .apply(merge_team_ids)
###################all_constructors["constructorId"] = all_constructors["constructorId"].apply(merge_team_ids)

#Constructors
current_constructors = ["mercedes","red_bull","mclaren","ferrari","alpine","aston_martin","rb","alfa","williams","haas"]

const_rates = []
for cost in current_constructors:
	constr_id = all_constructors.loc[all_constructors["constructorRef"] == cost, "constructorId"].item()
	#print(constr_id)
	const_all_rr = race_results.loc[race_results['constructorId'] == constr_id]
	const_no_errors = const_all_rr.loc[const_all_rr['statusId'].isin(constr_no_err_ids)]
	const_err_rate = (len(const_all_rr) - len(const_no_errors))/len(const_all_rr)
	const_rates.append(const_err_rate)
	#print(const_err_rate)

#Plot the error rates constrictors
fig, ax = plt.subplots()

ax.bar(current_constructors, const_rates)
ax.set_xlabel('Constructor ID')
ax.set_ylabel('Error Rate')
ax.set_title('Error Rate of all constructors')

plt.show()

#Find the correspodning contructorids for the current constructors
current_constructor_ids = []
for a_curr_constructor in current_constructors:
	constr_index = all_constructors.loc[all_constructors["constructorRef"] == a_curr_constructor, "constructorId"].item()
	current_constructor_ids.append(constr_index)
print(current_constructor_ids)

#Make a csv file with the error rates of all drivers but use the driver ids of the current drivers
driver_error_df = pd.DataFrame(list(zip(current_drivers, current_driver_rates)), columns =['code', 'DriverErrorRate'])
driver_error_df.to_csv("driver_error_rates.csv")

#Make a csv file with the error rates of all constructors using the constructor ids of the current constructors
constructor_error_df = pd.DataFrame(list(zip(current_constructor_ids, const_rates)), columns =['constructorId', 'ConstrErrorRate'])
constructor_error_df.to_csv("constructor_error_rates.csv")


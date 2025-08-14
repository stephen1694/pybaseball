# pybaseball
Exploring MLB statcast data with Python

make_MLB_ID_key.py  
This python script reads in people.csv files downloaded from https://github.com/chadwickbureau/register/tree/master/data, which contain baseball player and people names along with identification numbers from various websites like MLBAM, Fangraphs, and Baseball Reference. It concatenates names from all of these files into one dataframe, filteres for certain criteria, and writes out the resulting file MLBL_ID_key.csv.

download_2015_2024.py  
This python script downloads MLB Statcast data for each year between 2015-2024 using the pybaseball module in Python and outputs a .csv file. It then reads in certain columns from each year's dataframe and merges data for each year into one dataframe to write out. Last, it filters for pitches than ended with a ball in play and writes out a file with these data only.

predicting_hr_with_spray_angle.py
This python script uses the file created in download_2015_2024.py (which contains statcast information for each plate appearance of the 2015-2024 statcast era) and Savant_statcast_stats.csv (which contains barrel and plate appearance totals for each player year comobination, min. 300 PA). It uses exit velocity, launch angle, and spray angle batted ball information to create a distribution that reflects hr probability of each batted ball using these three variables, and attempts to outperform barrel rate in predicting future home run rate.

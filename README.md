# pybaseball
Exploring MLB statcast data with Python

make_MLB_ID_key.py  
This python script reads in people.csv files downloaded from https://github.com/chadwickbureau/register/tree/master/data, which contain baseball player and people names along with identification numbers from various websites like MLBAM, Fangraphs, and Baseball REference. It concatenates names from all of these files into one dataframe, filteres for certain criteria, and writes out the resulting file MLBL_ID_key.csv.

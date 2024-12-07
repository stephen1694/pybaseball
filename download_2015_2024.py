## step 1: download pitch level data for each year

# import modules
from pybaseball import statcast
import pandas as pd
from calendar import monthrange

def download_statcast_data(year, start_day, end_day):
    """
    Download Statcast data for a given year, starting and ending at specific dates.

    Parameters:
    - year: String, last two digits of the year (e.g., '15' for 2015)
    - start_day: String, the starting day in 'MM-DD' format (e.g., '03-15' for March 15)
    - end_day: String, the ending day in 'MM-DD' format (e.g., '10-05' for October 5)

    Returns finalDf: DataFrame with Statcast data for the specified range of the season
    """
    # Convert start and end days to full dates
    start_date = f"20{year}-{start_day}"
    end_date = f"20{year}-{end_day}"

    # Extract start and end months
    start_month = int(start_day.split('-')[0])
    end_month = int(end_day.split('-')[0])

    # Generate the list of months between the start and end
    list_months = [f"{month:02d}" for month in range(start_month, end_month + 1)]

    # Make an empty list for dataframes
    list_dfs = []

    # Download Statcast data month by month
    for i, month in enumerate(list_months):
        # Define start and end dates for the month
        if i == 0:  # First month
            start = start_date
            _, last_day = monthrange(int(f"20{year}"), int(month))
            end = f"20{year}-{month}-{last_day:02d}" if len(list_months) > 1 else end_date
        elif i == len(list_months) - 1:  # Last month
            start = f"20{year}-{month}-01"
            end = end_date
        else:  # Middle months
            start = f"20{year}-{month}-01"
            _, last_day = monthrange(int(f"20{year}"), int(month))
            end = f"20{year}-{month}-{last_day:02d}"

        print(f"Downloading data from {start} to {end}")
        data = statcast(start_dt=start, end_dt=end)
        list_dfs.append(data)

    # Concatenate dataframes in list_dfs to one dataframe called finalDf
    finalDf = pd.concat(list_dfs)

    # Drop duplicates from finalDf
    finalDf = finalDf.drop_duplicates()

    return finalDf

pitch_2015_df = download_statcast_data('15', '04-05', '10-04')
pitch_2015_df.to_csv('pitch_2015.csv')

#pitch_2016_df = download_statcast_data('16', '04-03', '10-02')
#pitch_2016_df = pitch_2016_df[~((pitch_2016_df['game_date'] == '2016-04-03') & (~pitch_2016_df['home_team'].isin(['PIT', 'KC', 'TB'])))] # drop spring games ((all except PIT, KC, TB home games)
#pitch_2016_df.to_csv('pitch_2016.csv')

#pitch_2017_df = download_statcast_data('17', '04-02', '10-01')
#pitch_2017_df.to_csv('pitch_2017.csv')

#pitch_2018_df = download_statcast_data('18', '03-29', '10-01')
#pitch_2018_df.to_csv('pitch_2018.csv')

#pitch_2019_df = download_statcast_data('19', '03-20', '09-30')
#pitch_2019_df = pitch_2019_df[~((pitch_2019_df['game_date'] == '2019-03-20') & (~pitch_2019_df['home_team'].isin(['SEA', 'OAK'])))] # drop non Japan series games 3/20
#pitch_2019_df = pitch_2019_df[~((pitch_2019_df['game_date'] == '2019-03-21') & (~pitch_2019_df['home_team'].isin(['SEA', 'OAK'])))] # drop non Japan series games 3/21
#dates_to_drop_19 = ['2019-03-22','2019-03-23','2019-03-24','2019-03-25','2019-03-26','2019-03-27'] # drop all games 3/22-3/27
#pitch_2019_df = pitch_2019_df[~pitch_2019_df['game_date'].isin(dates_to_drop_19)]
#pitch_2019_df.to_csv('pitch_2019.csv')

#pitch_2020_df = download_statcast_data('20', '07-23', '09-27')
#pitch_2020_df.to_csv('pitch_2020.csv')

#pitch_2021_df = download_statcast_data('21', '04-01', '10-03')
#pitch_2021_df.to_csv('pitch_2021.csv')

#pitch_2022_df = download_statcast_data('22', '04-07', '10-05')
#pitch_2022_df.to_csv('pitch_2022.csv')

#pitch_2023_df = download_statcast_data('23', '03-30', '10-01')
#pitch_2023_df.to_csv('pitch_2023.csv')

#pitch_2024_df = download_statcast_data('24', '03-20', '09-30')
#pitch_2024_df = pitch_2024_df[~((pitch_2024_df['game_date'] == '2024-03-20') & (~pitch_2024_df['home_team'].isin(['LAD', 'SD'])))] # drop non Seoul series games 3/20
#pitch_2024_df = pitch_2024_df[~((pitch_2024_df['game_date'] == '2024-03-21') & (~pitch_2024_df['home_team'].isin(['LAD', 'SD'])))] # drop non Seoul series games 3/21
#dates_to_drop_24 = ['2024-03-22','2024-03-23', '2024-03-24','2024-03-25', '2024-03-26', '2024-03-27'] # drop all games 3/22-3/27
#pitch_2024_df = pitch_2024_df[~pitch_2024_df['game_date'].isin(dates_to_drop_24)]
#pitch_2024_df.to_csv('pitch_2024.csv')


## step 2: load in, subset for certain columns, concatonate years into 1 df, and save this final df as pitch_2021-2024.csv
def load_df_csv(file_name, df_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # subset cols to keep (explained https://baseballsavant.mlb.com/csv-docs)
    df = df[['batter', 'game_year', 'game_date', 'home_team', 'stand', 'p_throws',
             'pitch_type', 'effective_speed', 'pfx_x', 'pfx_z', 'zone', 'description',
             'launch_speed', 'launch_angle', 'hc_x', 'hc_y', 'if_fielding_alignment', 'events']]

    return df

pitch_2015_df = load_df_csv("pitch_2015.csv", "pitch_2015_df")
pitch_2016_df = load_df_csv("pitch_2016.csv", "pitch_2016_df")
pitch_2017_df = load_df_csv("pitch_2017.csv", "pitch_2017_df")
pitch_2018_df = load_df_csv("pitch_2018.csv", "pitch_2018_df")
pitch_2019_df = load_df_csv("pitch_2019.csv", "pitch_2019_df")
pitch_2020_df = load_df_csv("pitch_2020.csv", "pitch_2020_df")
pitch_2021_df = load_df_csv("pitch_2021.csv", "pitch_2021_df")
pitch_2022_df = load_df_csv("pitch_2022.csv", "pitch_2022_df")
pitch_2023_df = load_df_csv("pitch_2023.csv", "pitch_2023_df")
pitch_2024_df = load_df_csv("pitch_2024.csv", "pitch_2024_df")

# concatonate dfs into one
batterStatcast = pd.concat([pitch_2015_df, pitch_2016_df, pitch_2017_df, pitch_2018_df, pitch_2019_df, pitch_2020_df, pitch_2021_df, pitch_2022_df, pitch_2023_df, pitch_2024_df], axis=0, ignore_index=True) # axis = 0 to bind by row rather than by col
print(batterStatcast.info())

# write out pitch level data for 2015-24
batterStatcast.to_csv('pitch_2015-2024.csv')

# filter for pitches hit into play and inspect
batterStatcast = batterStatcast[batterStatcast['description'].str.contains('hit_into_play')]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(batterStatcast.info())
print(batterStatcast.head(30))

# write out batted ball data for 2015-24
batterStatcast.to_csv('BIP_2015-2024.csv')

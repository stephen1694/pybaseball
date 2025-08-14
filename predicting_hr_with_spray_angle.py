# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import PA level 2015-2024 df
batterStatcast = pd.read_csv("PA_2015-2024.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(batterStatcast.info())
print(batterStatcast.head())

# calculate X and Y locations from hc_x and hc_y
# described here: https://tht.fangraphs.com/research-notebook-new-format-for-statcast-data-export-at-baseball-savant/
# and used here with rounded values and rearranged formula: https://visualbaseballinfo.blogspot.com/2017/06/spray-angle-horizontal-angle-for.html?source=post_page-----65a4b5d63f23--------------------------------
batterStatcast['X'] = batterStatcast['hc_x'] - 125.42
batterStatcast['Y'] = 198.72 - batterStatcast['hc_y']

# calculate spray angle
batterStatcast['spray_angle'] = (np.arctan(batterStatcast['X'] / batterStatcast['Y']) * 180 / np.pi * 0.75).round(1)
print(batterStatcast.info())

# plot all and just RHB (looks good- more deep hr to pull side confirming we are on right track)
plt.scatter(batterStatcast['X'], batterStatcast['Y'], s=5)
plt.show()
plt.scatter(batterStatcast[batterStatcast['stand'] == 'R']['X'], batterStatcast[batterStatcast['stand'] == 'R']['Y'], s=5)
plt.show()

## assign spray_field accounting for handedness

# Create a function to determine the appropriate labels based on 'stand'
def get_spray_field(row, bins_left, labels_left, bins_right, labels_right):
    if row['stand'] == 'R':
        return pd.cut([row['spray_angle']], bins=bins_right, labels=labels_right)[0]
    elif row['stand'] == 'L':
        return pd.cut([row['spray_angle']], bins=bins_left, labels=labels_left)[0]
    else:
        return None  # Handle cases where stand is neither 'R' nor 'L'

# Define the bins and labels for right and left stands
bins_left = [-float('inf'), -45, -15, 15, 45, float('inf')]
labels_left = ['oppo_foul', 'oppo', 'center', 'pull', 'pull_foul']

bins_right = [-float('inf'), -45, -15, 15, 45, float('inf')]
labels_right = ['pull_foul', 'pull', 'center', 'oppo', 'oppo_foul']

# Apply the function to create the 'spray_field' column
batterStatcast['spray_field'] = batterStatcast.apply(
    get_spray_field,
    axis=1,
    bins_left=bins_left,
    labels_left=labels_left,
    bins_right=bins_right,
    labels_right=labels_right
)
print(batterStatcast.head())

# Delete the temporary variables
del bins_left, labels_left, bins_right, labels_right

# make is_home_run col & convert to integer
batterStatcast['is_home_run'] = batterStatcast['events'] == 'home_run'
batterStatcast['is_home_run'] = batterStatcast['is_home_run'].astype(int)
print(batterStatcast.head())

# get average exit velo of hr overall & by field (launch speed filter added to remove likely inside the park hr)
print(batterStatcast[(batterStatcast['events'] == 'home_run') & (batterStatcast['launch_speed'] >= 90)]['launch_angle'].mean())
print(batterStatcast[(batterStatcast['events'] == 'home_run') & (batterStatcast['launch_speed'] >= 90) & (batterStatcast['spray_field'] == 'pull')]['launch_angle'].mean())
print(batterStatcast[(batterStatcast['events'] == 'home_run') & (batterStatcast['launch_speed'] >= 90) & (batterStatcast['spray_field'] == 'center')]['launch_angle'].mean())
print(batterStatcast[(batterStatcast['events'] == 'home_run') & (batterStatcast['launch_speed'] >= 90) & (batterStatcast['spray_field'] == 'oppo')]['launch_angle'].mean())

# calculate difference of spray angle from pull field foul pole
for index, row in batterStatcast.iterrows():
    if row['stand'] == 'R':
        batterStatcast.at[index, 'spray_angle'] += 45
    elif row['stand'] == 'L':
        batterStatcast.at[index, 'spray_angle'] = abs(batterStatcast.at[index, 'spray_angle']) + 45
    else:
        batterStatcast.at[index, 'spray_angle'] = np.nan  # Use NaN for missing or undefined 'stand' values

# calculate absolute value of difference from optimal launch angle by spray_field
for index, row in batterStatcast.iterrows():
    if row['spray_field'] == 'center':
        batterStatcast.at[index, 'diff_optimal_angle'] = abs(28 - batterStatcast.at[index, 'launch_angle'])
    elif row['spray_field'] == 'pull':
        batterStatcast.at[index, 'diff_optimal_angle'] = abs(29 - batterStatcast.at[index, 'launch_angle'])
    elif row['spray_field'] == 'oppo':
        batterStatcast.at[index, 'diff_optimal_angle'] = abs(29 - batterStatcast.at[index, 'launch_angle'])
    else:
        batterStatcast.at[index, 'spray_angle'] = np.nan  # Use NaN for missing or undefined 'stand' values
print(batterStatcast.head())

# Creating 'launch_speed_rounded' column rounded to nearest degree
batterStatcast['launch_speed_rounded'] = batterStatcast['launch_speed'].round(0)
print(batterStatcast.head())

# Group by 'launch_speed_rounded' and 'launch_angle' and calculate home run probability for each combo
home_run_prob = batterStatcast.groupby(['launch_speed_rounded', 'launch_angle'])['is_home_run'].mean().reset_index()

# Rename the 'is_home_run' column
home_run_prob.rename(columns={'is_home_run': 'home_run_probability'}, inplace=True)
print(home_run_prob.head())


## function to plot hr prob by angle & velo

import seaborn as sns

def plot_home_run_probability_heatmap(dataframe, launch_angle_range=(10, 50), launch_speed_range=(90, 120)):
    """
    Plots a heatmap of home run probability based on launch angle and launch speed.

    Parameters:
    - dataframe: The DataFrame containing the necessary data.
    - launch_angle_range: A tuple specifying the range of launch angles to filter.
    - launch_speed_range: A tuple specifying the range of launch speeds to filter.
    """
    # Group by 'launch_speed_rounded' and 'launch_angle' and calculate home run probability
    home_run_prob = dataframe.groupby(['launch_speed_rounded', 'launch_angle'])['is_home_run'].mean().reset_index()

    # Rename the 'is_home_run' column to represent probability
    home_run_prob.rename(columns={'is_home_run': 'home_run_probability'}, inplace=True)

    # Print the first few rows of the grouped DataFrame
    print(home_run_prob.head())

    # Pivot the data to create a matrix suitable for a heatmap
    heatmap_data = home_run_prob.pivot_table(index='launch_angle', columns='launch_speed_rounded',
                                             values='home_run_probability')

    # Filter data to zoom in on specific x and y ranges
    heatmap_data = heatmap_data.loc[launch_angle_range[0]:launch_angle_range[1],
                   launch_speed_range[0]:launch_speed_range[1]]

    # Reverse y-axis
    heatmap_data = heatmap_data[::-1]

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, cbar_kws={'label': 'Home Run Probability'}, linewidths=0.5,
                yticklabels=True)

    # Set titles and labels
    plt.title('Home Run Probability Heatmap (Zoomed In)')
    plt.xlabel('Launch Speed (Rounded)')
    plt.ylabel('Launch Angle')

    # Set x & y axis tick marks to integer values, centered over heatmap blocks
    plt.yticks(ticks=np.arange(len(heatmap_data.index)) + 0.5, labels=[int(row) for row in heatmap_data.index],
               rotation=0)
    plt.xticks(ticks=np.arange(len(heatmap_data.columns)) + 0.5, labels=[int(col) for col in heatmap_data.columns],
               rotation=0)

    # Display the plot
    plt.show()


# call function
plot_home_run_probability_heatmap(batterStatcast)
plot_home_run_probability_heatmap(batterStatcast[batterStatcast['spray_field'] == "pull"])
plot_home_run_probability_heatmap(batterStatcast[batterStatcast['spray_field'] == "center"])
plot_home_run_probability_heatmap(batterStatcast[batterStatcast['spray_field'] == "oppo"])

# check- should be higher for pulled than oppo :)
print(batterStatcast[batterStatcast['spray_field'] == 'pull']['launch_speed'].mean())
print(batterStatcast[batterStatcast['spray_field'] == 'center']['launch_speed'].mean())
print(batterStatcast[batterStatcast['spray_field'] == 'oppo']['launch_speed'].mean())

# write out this modified 2015-2024 PA file
batterStatcast.to_csv('PA_2015-2024_modified.csv')






## Now start quantifying hr probability with exit velo, launch angle, & spray angle

# import previous modules
import pandas as pd
import numpy as np
import seaborn as sns

# import modified 2015-2024 PA file
batterStatcast = pd.read_csv("PA_2015-2024_modified.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(batterStatcast.head())

# make spray_angle_rounded rounded to nearest 10 degrees
batterStatcast['spray_angle_rounded'] = batterStatcast['spray_angle'].apply(lambda x: round(x / 10) * 10 if pd.notna(x) else np.nan)
print(batterStatcast.head())

# new modules
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# function that plots hr probability in 3D space using gaussian smoothing
def plot_home_run_probability_3d(dataframe, launch_angle_range=(10, 50), launch_speed_range=(90, 120)):
    """
    Plots a 3D scatter plot of home run probability based on launch angle, launch speed, and spray angle (spray_field5),
    with smoothed probability values.

    Parameters:
    - dataframe: The DataFrame containing the necessary data.
    - launch_angle_range: A tuple specifying the range of launch angles to filter.
    - launch_speed_range: A tuple specifying the range of launch speeds to filter.
    """
    # Filter data based on given ranges
    filtered_data = dataframe[
        (dataframe['launch_angle'] >= launch_angle_range[0]) &
        (dataframe['launch_angle'] <= launch_angle_range[1]) &
        (dataframe['launch_speed'] >= launch_speed_range[0]) &
        (dataframe['launch_speed'] <= launch_speed_range[1])
    ]

    # Group by 'launch_speed_rounded', 'launch_angle', and 'spray_angle_rounded' to calculate home run probability
    home_run_prob = filtered_data.groupby(['launch_speed_rounded', 'launch_angle', 'spray_angle_rounded'])[
        'is_home_run'].mean().reset_index()
    home_run_prob.rename(columns={'is_home_run': 'home_run_probability'}, inplace=True)

    # Filter out rows with a home run probability of < 10%
    home_run_prob = home_run_prob[home_run_prob['home_run_probability'] > .1]

    # Define the grid range
    grid_launch_speed = np.linspace(launch_speed_range[0], launch_speed_range[1], 50)
    grid_launch_angle = np.linspace(launch_angle_range[0], launch_angle_range[1], 50)
    grid_spray_angle = np.linspace(0, 90, 50)
    grid_x, grid_y, grid_z = np.meshgrid(grid_launch_speed, grid_launch_angle, grid_spray_angle)

    # Interpolate probabilities on the grid
    grid_prob = griddata(
        (home_run_prob['launch_speed_rounded'],
         home_run_prob['launch_angle'],
         home_run_prob['spray_angle_rounded']),
        home_run_prob['home_run_probability'],
        (grid_x, grid_y, grid_z),
        method='linear'
    )

    # Apply Gaussian smoothing to the probability grid (optional)
    grid_prob = gaussian_filter(grid_prob, sigma=1)

    # 3D plot setup with improved aesthetics
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Scatter plot the smoothed probabilities
    sc = ax.scatter(
        grid_x.flatten(),
        grid_y.flatten(),
        grid_z.flatten(),
        c=grid_prob.flatten(),
        cmap='coolwarm',
        s=10,
        alpha=0.7,
        edgecolor='k'
    )

    # Set axis labels
    ax.set_xlabel('Launch Speed (Rounded)', labelpad=15, fontsize=12, fontweight='bold')
    ax.set_ylabel('Launch Angle', labelpad=15, fontsize=12, fontweight='bold')
    ax.set_zlabel('Spray Angle', labelpad=15, fontsize=12, fontweight='bold')
    ax.set_title('Smoothed 3D Home Run Probability Based on Launch Speed, Angle, and Spray Angle', fontsize=14,
                 fontweight='bold', pad=20)

    # Set x-axis (launch speed) and y-axis (launch angle) to ascending order by swapping min and max values
    ax.set_xlim(launch_speed_range[1], launch_speed_range[0])  # Flipped order
    ax.set_ylim(launch_angle_range[1], launch_angle_range[0])  # Flipped order

    # Remove grid lines and add a clean background
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Add color bar to indicate home run probability
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Home Run Probability', fontsize=12, fontweight='bold')

    # Rotate the view for a better perspective
    ax.view_init(elev=20, azim=120)

    # Show the plot
    plt.show()


# Call the function
plot_home_run_probability_3d(batterStatcast, launch_angle_range=(10, 50), launch_speed_range=(90, 120))

## use calculated probabilities in plot to get probability of each batted ball being hr

from scipy.interpolate import RegularGridInterpolator

def assign_home_run_probabilities(dataframe, launch_angle_range=(10, 50), launch_speed_range=(90, 120)):
    """
    Plots a 3D scatter plot of home run probability based on launch angle, launch speed, and spray angle,
    with smoothed probability values, and assigns home run probability to each row in the DataFrame.

    Parameters:
    - dataframe: The DataFrame containing the necessary data.
    - launch_angle_range: A tuple specifying the range of launch angles to filter.
    - launch_speed_range: A tuple specifying the range of launch speeds to filter.
    """
    # Filter data based on given ranges
    filtered_data = dataframe[
        (dataframe['launch_angle'] >= launch_angle_range[0]) &
        (dataframe['launch_angle'] <= launch_angle_range[1]) &
        (dataframe['launch_speed'] >= launch_speed_range[0]) &
        (dataframe['launch_speed'] <= launch_speed_range[1])
    ]

    # Group by 'launch_speed_rounded', 'launch_angle', and 'spray_angle_rounded' to calculate home run probability
    home_run_prob = filtered_data.groupby(['launch_speed_rounded', 'launch_angle', 'spray_angle_rounded'])[
        'is_home_run'].mean().reset_index()
    home_run_prob.rename(columns={'is_home_run': 'home_run_probability'}, inplace=True)

    # Define the grid range
    grid_launch_speed = np.linspace(launch_speed_range[0], launch_speed_range[1], 50)
    grid_launch_angle = np.linspace(launch_angle_range[0], launch_angle_range[1], 50)
    grid_spray_angle = np.linspace(0, 90, 50)
    grid_x, grid_y, grid_z = np.meshgrid(grid_launch_speed, grid_launch_angle, grid_spray_angle)

    # Interpolate probabilities on the grid
    grid_prob = griddata(
        (home_run_prob['launch_speed_rounded'],
         home_run_prob['launch_angle'],
         home_run_prob['spray_angle_rounded']),
        home_run_prob['home_run_probability'],
        (grid_x, grid_y, grid_z),
        method='linear'
    )

    # Apply Gaussian smoothing to the probability grid
    grid_prob = gaussian_filter(grid_prob, sigma=1)

    # Create an interpolator to assign home run probability to each batted ball
    interpolator = RegularGridInterpolator(
        (grid_launch_speed, grid_launch_angle, grid_spray_angle),
        grid_prob,
        bounds_error=False,
        fill_value=0
    )

    # Assign probabilities to each row in the original DataFrame
    dataframe['home_run_probability'] = interpolator(
        dataframe[['launch_speed', 'launch_angle', 'spray_angle']].values
    )

    # Return the updated DataFrame with the new column
    return dataframe


# Call function
batterStatcast = assign_home_run_probabilities(batterStatcast, launch_angle_range=(10, 50), launch_speed_range=(90, 120))

# Display updated DataFrame with home run probabilities
print(batterStatcast.head())

# make likely_hr col & convert to integer
batterStatcast['nonzero_HR'] = batterStatcast['home_run_probability'] > 0
batterStatcast['nonzero_HR'] = batterStatcast['nonzero_HR'].astype(int)
batterStatcast['25pct_HR'] = batterStatcast['home_run_probability'] >= 0.25
batterStatcast['25pct_HR'] = batterStatcast['25pct_HR'].astype(int)
batterStatcast['50pct_HR'] = batterStatcast['home_run_probability'] >= 0.5
batterStatcast['50pct_HR'] = batterStatcast['50pct_HR'].astype(int)
batterStatcast['75pct_HR'] = batterStatcast['home_run_probability'] >= 0.75
batterStatcast['75pct_HR'] = batterStatcast['75pct_HR'].astype(int)
batterStatcast['95pct_HR'] = batterStatcast['home_run_probability'] >= 0.95
batterStatcast['95pct_HR'] = batterStatcast['95pct_HR'].astype(int)
print(batterStatcast.head())

# Define columns to average and sum
columns_to_average = ['launch_speed', 'diff_optimal_angle', 'spray_angle']
columns_to_sum = ['is_home_run', 'home_run_probability', 'nonzero_HR', '25pct_HR', '50pct_HR', '75pct_HR', '95pct_HR']

# Define the aggregation dictionary
aggregation_dict = {col: 'mean' for col in columns_to_average}
aggregation_dict.update({col: 'sum' for col in columns_to_sum})

# Group by batter and game_year with specified aggregations
grouped_df = batterStatcast.groupby(['batter', 'game_year']).agg(aggregation_dict).reset_index()

# Calculate occurrences separately and merge it back
occurrences = batterStatcast.groupby(['batter', 'game_year']).size().reset_index(name='occurrences')

# Merge occurrences into grouped_df
grouped_df = grouped_df.merge(occurrences, on=['batter', 'game_year'])

# Display the result
print(grouped_df.head())

# Sort grouped_df by batter and year
grouped_df = grouped_df.sort_values(['batter', 'game_year']).reset_index(drop=True)

# Initialize next_yr_hr
grouped_df['next_yr_hr'] = pd.NA

# Loop through each row to get a players next_yr_hr total if the next row is the same batter and the following year
for i in range(len(grouped_df) - 1):
    if grouped_df.loc[i, 'batter'] == grouped_df.loc[i + 1, 'batter'] and \
       grouped_df.loc[i + 1, 'game_year'] == grouped_df.loc[i, 'game_year'] + 1:
        grouped_df.loc[i, 'next_yr_hr'] = int(grouped_df.loc[i + 1, 'is_home_run'])

# Convert to integer
grouped_df['next_yr_hr'] = grouped_df['next_yr_hr'].astype(pd.Int64Dtype())
print(grouped_df.head())

# Load Statcast data
StatcastSeasons = pd.read_csv("Savant_statcast_stats.csv")
print(StatcastSeasons.head())

# Rename columns for merging
grouped_df = grouped_df.rename(columns={'game_year': 'year', 'batter': 'player_id'})
StatcastSeasons = StatcastSeasons.rename(columns={'last_name, first_name': 'name','player_age': 'age'})

# Select columns needed for merge
StatcastSeasons_filtered = StatcastSeasons[['player_id', 'name', 'year', 'age',
                                             'barrel', 'home_run', 'woba', 'xwoba', 'pa']]

# Merge with grouped_df_filtered
merged_df = grouped_df.merge(StatcastSeasons_filtered, on=['player_id', 'year'], how='left')
print(merged_df.head())

# Check if 'is_home_run' counts match Statcast 'home_run' total
merged_df['match'] = merged_df['is_home_run'] == merged_df['home_run']
match_counts = merged_df.groupby('year')['match'].value_counts().unstack(fill_value=0)
print(match_counts)

# Loop through each row to get a players next_yr_pa total if the next row is the same batter and the following year
merged_df = merged_df.sort_values(['player_id', 'year'])
merged_df['next_yr_pa'] = merged_df.groupby('player_id')['pa'].shift(-1)
mask = merged_df['year'] + 1 == merged_df.groupby('player_id')['year'].shift(-1)
merged_df.loc[~mask, 'next_yr_pa'] = np.nan
print(merged_df.head(10))

# Filter for rows at least 300 pa in the current and following year for analysis
merged_df2 = merged_df[(merged_df['pa'] >= 300) & (merged_df['next_yr_pa'] >= 300)]
merged_df = merged_df2
print(merged_df.head())



## test relative ranking of association with next year hr

# standardize
stats_to_standardize = ['barrel', 'home_run', 'home_run_probability','nonzero_HR', '25pct_HR', '50pct_HR', '75pct_HR', '95pct_HR']
for col in stats_to_standardize:
    merged_df[f'{col}_per_pa'] = merged_df[col] / merged_df['pa']
merged_df['next_yr_hr_per_pa'] = merged_df['next_yr_hr'] / merged_df['next_yr_pa']
print(merged_df.head())

# define var to test
variables = [
    'launch_speed', 'diff_optimal_angle', 'spray_angle',
    'home_run_per_pa', 'home_run_probability_per_pa', 'barrel_per_pa', 'nonzero_HR_per_pa', '25pct_HR_per_pa', '50pct_HR_per_pa', '75pct_HR_per_pa', '95pct_HR_per_pa'
]

# Calculate and sort correlations
correlations = merged_df[variables + ['next_yr_hr_per_pa']].corr()['next_yr_hr_per_pa'].drop('next_yr_hr_per_pa').sort_values(ascending=False)
correlations_abs = correlations.abs().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations_abs.index, y=correlations_abs.values)
plt.title('Correlation of Variables with next_yr_hr_per_pa')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# barrels is better than any of the other variables I calculated & tested

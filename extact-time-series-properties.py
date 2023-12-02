# Task 1:
#   Determine the time stamp when Auto mode starts.
#   The data starts with manual mode and continue until get message "AUTO MODE ACTIVE PLGM OFF"
#   in column "Q" ('Alarm')of the InsulinData.csv. Then Auto mode starts, no switch back.
# Task 2:
#   Figure out the timestamp in CGMData.csv where Auto mode starts.
#   search for the timestamp that is nearest to (and later than) the Auto mode start time.
# Task 7:
#   Dealing with missing data and duplicates
#   Delete of the entire day of data. Or Interpolation.
# check how many days contained missing value
# na_data = CGM_df[CGM_df['Sensor Glucose (mg/dL)'].isna()]
# print(na_data['Date'].nunique(), CGM_df['Date'].nunique(), na_data)
# print(CGM_df.duplicated().value_counts()) # No duplicates
# 158 days out of 195 days has missing value, can't drop the entire day.
# print(CGM_df['Sensor Glucose (mg/dL)'].mean()), mean value is 147, can't interpolate with mean
# Directly dropna.
# Task 3:
#   Parse and divide CGM data into segments of a day. 00:00-23:59
# Task 4:
#   Sub-segments: daytime 06:00-23:59, overnight: 00:00-05:59
# Task 5:
#   Count the number of samples that belong to the ranges specified in the metrics.
# Task 6:
#   Calculate the percentage
#   tackle with both Manual mode and Auto mode
# Task 8:
#   Save the result in Result.csv

import pandas as pd
import numpy as np


MODECHANGE = "AUTO MODE ACTIVE PLGM OFF"


def calculation(df):
    """Calculate the sum of times"""
    if len(df) == 0:
        return 0
    else:
        return (df.groupby('Date')['Date'].count()).sum()


def get_metrics(df):
    """return a list of six metrics"""
    m_h = calculation(df[df['Sensor Glucose (mg/dL)'] > 180])
    m_h_c = calculation(df[df['Sensor Glucose (mg/dL)'] > 250])
    m_r = calculation(df[(df['Sensor Glucose (mg/dL)'] > 70) & (df['Sensor Glucose (mg/dL)'] < 180)])
    m_r_s = calculation(df[(df['Sensor Glucose (mg/dL)'] > 70) & (df['Sensor Glucose (mg/dL)'] < 150)])
    m_h_l1 = calculation(df[df['Sensor Glucose (mg/dL)'] < 70])
    m_h_l2 = calculation(df[df['Sensor Glucose (mg/dL)'] < 54])
    return [m_h, m_h_c, m_r, m_r_s, m_h_l1, m_h_l2]


# Load the files, check the basic info, generate timestamp
data1 = pd.read_csv("InsulinData.csv")
Insulin_df = data1[['Date', 'Time', 'Alarm']]
Insulin_df["Timestamp"] = pd.to_datetime(Insulin_df['Date'] + ' ' + Insulin_df['Time'])
data2 = pd.read_csv("CGMData.csv")
CGM_df = data2[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
CGM_df['Timestamp'] = pd.to_datetime(CGM_df['Date'] + ' ' + CGM_df['Time'])
CGM_df = CGM_df.dropna()  # dropna.

# find the mode changing time in both Insulin_df and CGM_df, create auto_df and manual_df in three time interval
changing_time = Insulin_df.loc[Insulin_df['Alarm'] == MODECHANGE]['Timestamp'].min()

wholeday_auto_df = CGM_df.loc[CGM_df['Timestamp'] > changing_time]
wholeday_manual_df = CGM_df.loc[CGM_df['Timestamp'] <= changing_time]
auto_dates = wholeday_auto_df["Date"].nunique()
manual_dates = wholeday_manual_df["Date"].nunique()

overnight_auto_df = wholeday_auto_df.loc[wholeday_auto_df['Timestamp'].dt.hour.between(0, 5)]
overnight_manual_df = wholeday_manual_df.loc[wholeday_manual_df['Timestamp'].dt.hour.between(0, 5)]
daytime_auto_df = wholeday_auto_df.loc[wholeday_auto_df['Timestamp'].dt.hour.between(6, 23)]
daytime_manual_df = wholeday_manual_df.loc[wholeday_manual_df['Timestamp'].dt.hour.between(6, 23)]

# metrics in list
manual_metrics_sum = (get_metrics(overnight_manual_df) + get_metrics(daytime_manual_df) + get_metrics(wholeday_manual_df))
auto_metrics_sum = get_metrics(overnight_auto_df) + get_metrics(daytime_auto_df) + get_metrics(wholeday_auto_df)
manual_metrics = [round(i * 100 / manual_dates / 288, 2) for i in manual_metrics_sum]
auto_metrics = [round(i * 100 / auto_dates / 288, 2) for i in auto_metrics_sum]

result = pd.DataFrame(data=[manual_metrics, auto_metrics],
                      index=["Manual Mode", "Auto Mode"],
                      columns=["Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)",
                               "Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
                               "Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
                               "Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
                               "Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
                               "Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
                               "Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)",
                               "Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
                               "Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
                               "Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
                               "Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
                               "Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
                               "Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)",
                               "Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
                               "Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
                               "Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
                               "Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
                               "Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"])

result.to_csv("Result.csv", index=False, header=None)

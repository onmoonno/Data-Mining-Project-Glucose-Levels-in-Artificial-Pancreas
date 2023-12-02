import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN



def read_insulin(df):
    insulin_df = df[['Date', 'Time', 'BWZ Carb Input (grams)']].copy()
    insulin_df["Timestamp"] = pd.to_datetime(insulin_df['Date'] + ' ' + insulin_df['Time'])
    insulin_df.set_index("Timestamp", inplace=True)

    # only preserve the possible meal-time
    insulin_df['BWZ Carb Input (grams)'] = insulin_df['BWZ Carb Input (grams)'].replace(0, np.NaN)
    insulin_df.dropna(inplace=True)
    insulin_df.sort_values("Timestamp", inplace=True)
    return insulin_df


def read_cgm(df):
    cgm = df[['Date', 'Time', 'Sensor Glucose (mg/dL)']].copy()
    cgm['Timestamp'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'])
    cgm.set_index("Timestamp", inplace=True)
    cgm.sort_values("Timestamp", inplace=True)
    return cgm


def mealtime(df):
    """Take the insulin df, extract the list of start time of meal, already 30 min in advance. """
    insulin_df = read_insulin(df)
    # drop the timestamp of which the next timestamp is within 2 hours
    insulin_df['two_hours_later'] = insulin_df.index.shift(2, freq='H')
    insulin_df['Meal'] = 1
    for i, idx in enumerate(insulin_df.index):
        if i < len(insulin_df) - 1:
            if insulin_df.iloc[i, 3] >= insulin_df.index[i + 1]:
                insulin_df.iloc[i, 4] = np.NaN
                # print(insulin_df.index[i], insulin_df.index[i+1])  # another meal in 2 hours
    insulin_df.dropna(inplace=True)

    insulin_df['start meal'] = insulin_df.index.shift(-0.5, freq='H')
    insulin_df.set_index("start meal", inplace=True)
    stime = insulin_df.index.tolist()
    carb_input = insulin_df["BWZ Carb Input (grams)"].values.tolist()
    possible_meal = zip(stime, carb_input)
    return possible_meal


def extract_meal_data(df, meal_start_time):
    """Take a CGM DataFrame and start time list, return a DataFrame containing p * 30 meal data"""
    cgm = read_cgm(df)
    meal_df_column = np.array(range(0, 150, 5))
    meal_df = pd.DataFrame(columns=meal_df_column)
    carb_list = []
    for (time, carb) in meal_start_time:
        meal = cgm.loc[
            (cgm.index >= time) & (cgm.index <= time + pd.Timedelta('2.5 hours'))]
        glucose = meal['Sensor Glucose (mg/dL)'].values
        if len(glucose) != 30:
            continue
        meal_df.loc[len(meal_df)] = glucose
        carb_list.append(carb)
    meal_df["carb"] = carb_list
    meal_df = meal_df.dropna(axis=0)
    meal_df = meal_df.sort_values("carb").reset_index().drop("index", axis=1)
    return meal_df


def get_ground_truth(df):
    min_value = df["carb"].min()
    n = int((df["carb"].max() - df["carb"].min()) // 20)
    for i in range(n):
        df.loc[df["carb"].between(min_value + i * 20, min_value + (i + 1) * 20), "bin"] = i
    df = df.dropna(axis=0)
    return df


def cal_climb_up(df):
    """ Take the data df, return two features of the time series,
    including the climb up time span and dG(the time and value change
    from gluce min to max after meal."""
    feature = pd.DataFrame()
    feature["max_value"] = df.iloc[:, 2:20].max(axis=1)
    feature["max_time / min"] = df.iloc[:, 2:20].idxmax(axis=1)
    feature["min_value"] = df.iloc[:, 2:20].min(axis=1)
    feature["min_time /min"] = df.iloc[:, 2:20].idxmin(axis=1)
    feature["climb up time"] = feature["max_time / min"] - feature["min_time /min"]
    feature["dG"] = (feature["max_value"] - feature["min_value"]) / feature["min_value"]

    return feature['dG']


def cal_fft(df):
    fft_features = pd.DataFrame(columns=['pf'])
    for i in range(len(df)):
        # Extract the time series column as a numpy array
        data = df.iloc[i].values
        # Set the time step between measurements (in minutes)
        dt = 5
        # Apply FFT
        fft_data = np.fft.fft(data)
        freq = np.fft.fftfreq(len(data), dt)
        # calculate the power sd
        psd = np.abs(fft_data) ** 2
        # fetch the value and location of the  second and the third peak
        psd_list = [(psd_value, i) for (i, psd_value) in enumerate(psd)]
        psd_list.sort(reverse=True)
        pf1 = psd_list[0][0]
        pf2 = psd_list[1][0]
        f1_location = psd_list[0][1]
        f2_location = psd_list[1][1]
        f1 = freq[f1_location]
        f2 = freq[f2_location]
        f_feature = np.array([pf1])
        fft_features.loc[len(fft_features)] = f_feature
    return fft_features


def cal_diff(df):
    diff_feature = pd.DataFrame(columns=['diff1', 'diff2'])
    diff1_df = df.diff(axis=1)
    diff_feature['diff1'] = diff1_df.max(axis=1)
    diff_2_df = diff1_df.diff(axis=1)
    diff_feature['diff2'] = diff_2_df.max(axis=1)
    return diff_feature['diff2']


def cal_Entropy(df):
    entropy_df = pd.DataFrame(columns=["entropy"])
    for i in range(len(df)):
        # Extract the time series column as a numpy array
        data = df.iloc[i].values
        len_param = len(data)
        entropy = 0
        value, ctr = np.unique(data, return_counts=True)
        ratio = ctr / len_param
        ratio_nonzero = np.count_nonzero(ratio)
        if ratio_nonzero <= 1:
            entropy = 0
        for u in ratio:
            entropy -= u * np.log2(u)
        entropy_df.loc[len(entropy_df)] = entropy
    return entropy_df


def extract_features(df):
    climb_up = cal_climb_up(df)
    fft_features = cal_fft(df)
    diff_features = cal_diff(df)
    entropy = cal_Entropy(df)
    features = pd.concat([climb_up, fft_features, diff_features], axis=1)

    # Instantiate a StandardScaler object
    scaler = StandardScaler()
    # Fit the scaler to the DataFrame
    scaler.fit(features)
    # Transform the DataFrame
    features_std = scaler.transform(features)
    # Convert the standardized data back to a DataFrame
    features_df = pd.DataFrame(features_std, columns=features.columns)

    return features_df


def calculate_ep(df, cluster_name):
    """given the df and cluster name, which should be string,
       return the matrix used to calculate Entropy and Purity"""
    ep = pd.DataFrame()
    for bin in df['bin'].unique():
        df_ep = df.loc[df['bin'] == bin].groupby(cluster_name).count()['bin']
        ep = pd.concat([ep, df_ep], axis=1)
    ep = ep.replace(np.NaN, 0)
    ep = ep.sort_index()
    ep.columns = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4']

    sum_array = np.sum(ep.values, axis=1).reshape((5,1))
    weight = sum_array/len(df)
    weight_array = np.array(weight)
    freq_matrix = np.divide(ep.values, sum_array)
    eps = 1e-8
    freq_matrix = freq_matrix + eps
    entropy_list = []
    for row in freq_matrix:
        log_row = np.log2(row)
        entropy = -np.dot(row, log_row)
        entropy_list.append(entropy)
    entropy_array = np.array(entropy_list)

    ep['weights'] = weight_array
    ep['entropy'] = entropy_array
    ep['weighted_entropy'] = ep['weights']*ep['entropy']
    ep['max bin'] = ep.max(axis=1)
    total_entropy = round(sum(ep['weighted_entropy'].values), 2)
    total_purity = round(sum(ep['max bin'].values)/len(df), 2)
    return [total_entropy,total_purity]


# load data
insulin_df_1 = pd.read_csv("InsulinData.csv", low_memory=False)
cgm_df_1 = pd.read_csv("CGMData.csv", low_memory=False)

# extract meal data, generate ground truth, assign binnumbers
meal_time = mealtime(insulin_df_1)
meal_data = extract_meal_data(cgm_df_1, meal_time)
binned_meal = get_ground_truth(meal_data)
features = extract_features(binned_meal.iloc[:, :-2])


# Initialize k-means object with number of clusters
X = features
n = binned_meal["bin"].nunique()
kmeans = KMeans(n_clusters=n, random_state=42)
dbscan = DBSCAN()

# Fit the k-means object to the data
kmeans.fit(X)
dbscan.fit(X)

# Predict the cluster labels for each data point
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_

# Calculate the SSE_kmeans
sse_kmeans = kmeans.inertia_

# Calculate the SSE_dbscan, first find the centroids of the clusters
centroids = []
for label in np.unique(dbscan_labels):
    if label == -1:  # noise points are labeled as -1
        continue
    centroid = np.mean(X[dbscan_labels == label], axis=0)
    centroids.append(centroid)

# Calculate the SSE_dbscan
sse_dbscan = 0
for label in np.unique(dbscan_labels):
    if label == -1:
        continue
    cluster = X[dbscan_labels == label]
    centroid = centroids[label]
    sse_dbscan += np.sum((cluster - centroid) ** 2)
# print(sum(sse_dbscan), sse_kmeans)

# Combine the labels to calculate entropy and purity
features['bin'] = binned_meal['bin']
features['kmeans'] = kmeans_labels
features['dbscan'] = dbscan_labels

# generate the matrix and calculate entropy and purity
kmeans_result = calculate_ep(features, 'kmeans')
features['dbscan'] = features['dbscan'].replace(-1, np.NaN)
features = features.dropna(axis=0)
dbscan_result = calculate_ep(features, 'dbscan')
result = [sse_kmeans,sum(sse_dbscan), kmeans_result[0], dbscan_result[0], kmeans_result[1], dbscan_result[1]]
result_df = pd.DataFrame(data=result)
result_df = result_df.transpose()
result_df.to_csv('Result.csv',header=None, index=None)

print(features.info())
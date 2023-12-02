import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.feature_selection import RFE
import pickle


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
    return insulin_df.index.tolist()


def extract_meal_data(df, meal_start_time):
    """Take a CGM DataFrame and start time list, return a DataFrame containing p * 30 meal data"""
    cgm = read_cgm(df)
    meal_df_column = np.array(range(0, 120, 5))
    meal_df = pd.DataFrame(columns=meal_df_column)
    for i in range(1, len(meal_start_time)):
        meal = cgm.loc[
            (cgm.index >= meal_start_time[i]) & (cgm.index <= meal_start_time[i] + pd.Timedelta('2 hours'))]
        glucose = meal['Sensor Glucose (mg/dL)'].values
        if len(glucose) != 24:
            continue
        meal_df.loc[len(meal_df)] = glucose
    meal_df = meal_df.dropna(axis=0)
    return meal_df


def nomealtime(df):
    """take a insulin dataframe, return a list of (nomeal start time, nomeal end time),
    which the nomeal time > 2 hours """
    insulin_df = read_insulin(df)
    insulin_df["start meal"] = insulin_df.index.shift(-0.5, freq='H')
    insulin_df["end meal"] = insulin_df.index.shift(2, freq='H')
    no_meal = []
    for i in range(len(insulin_df) - 1):
        if (insulin_df['start meal'][i + 1] - insulin_df['end meal'][i]).total_seconds() <= 2 * 3600:
            continue  # expel the time period less than 2 hours
        no_meal.append((insulin_df['end meal'][i], insulin_df['start meal'][i + 1]))
    return no_meal


def extract_no_meal_data(df, nomeal_time_list):
    """Take cgm DataFrame and nomeal time list, return a q * 24 no meal dataframe"""
    cgm_df = read_cgm(df)
    nomeal_df_column = np.array(range(0, 120, 5))
    nomeal_df = pd.DataFrame(columns=nomeal_df_column)
    for (st, et) in nomeal_time_list:
        # num_of_possible_nomeals = int(((et - st).total_seconds()) // (2 * 3600))
        # for i in range(num_of_possible_nomeals):
        #     nomeal = cgm_df.loc[
        #         (cgm_df.index >= st + i * pd.Timedelta("2 hours")) & (
        #                     cgm_df.index <= st + (i + 1) * pd.Timedelta("2 hours"))]
        #     nomeals = nomeal['Sensor Glucose (mg/dL)'].values
        #     if len(nomeals) != 24:
        #         continue
        #     nomeal_df.loc[len(nomeal_df)] = nomeals
        nomeal = cgm_df.loc[
                    (cgm_df.index >= st )
                    & (cgm_df.index <= st + pd.Timedelta("2 hours"))]
        nomeals = nomeal['Sensor Glucose (mg/dL)'].values
        if len(nomeals) != 24:
            continue
        nomeal_df.loc[len(nomeal_df)] = nomeals
    nomeal_df = nomeal_df.dropna(axis=0)

    return nomeal_df.sample(200)


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
    fft_features = pd.DataFrame(columns=['pf1', 'f1', 'pf2','f2'])
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
        f_feature = np.array([pf1, f1, pf2, f2])
        fft_features.loc[len(fft_features)] = f_feature
    return fft_features


def cal_diff(df):
    diff_feature = pd.DataFrame(columns=['diff1', 'diff2'])
    diff1_df = df.diff(axis=1)
    diff_feature['diff1'] = diff1_df.max(axis=1)
    diff_2_df = diff1_df.diff(axis=1)
    diff_feature['diff2'] = diff_2_df.max(axis=1)
    return diff_feature


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
    features = pd.concat([climb_up, fft_features, diff_features, entropy], axis=1)
    return features


def model_training(df):
    """train and test svm and decision tree classifier with k fold cross validation, fold = 5
    save the two model in pickle format file, and print out the scores"""
    # Assume X is your feature matrix and y is your target vector
    X = df.iloc[:, :-1]
    y = df["class"]

    # Initialize SVM and Decision Tree classifiers
    svm_clf = SVC()
    dt_clf = DecisionTreeClassifier(min_samples_split=50, max_leaf_nodes=2)

    # Initialize performance metrics
    svm_f1_scores = []
    svm_precision_scores = []
    svm_accuracy_scores = []
    dt_f1_scores = []
    dt_precision_scores = []
    dt_accuracy_scores = []

    # Define 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate over the 5 folds
    for train_idx, test_idx in kfold.split(X, y):

        # Split the data into training and testing sets for the current fold
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # Fit SVM and Decision Tree classifiers on the training data for the current fold
        svm_clf.fit(X_train, y_train)
        dt_clf.fit(X_train, y_train)


        # Make predictions on the testing data for the current fold
        svm_preds = svm_clf.predict(X_test)
        dt_preds = dt_clf.predict(X_test)

        # Calculate F1 score, precision, and accuracy for SVM classifier on the current fold
        svm_f1_scores.append(f1_score(y_test, svm_preds))
        svm_precision_scores.append(precision_score(y_test, svm_preds))
        svm_accuracy_scores.append(accuracy_score(y_test, svm_preds))

        # Calculate F1 score, precision, and accuracy for Decision Tree classifier on the current fold
        dt_f1_scores.append(f1_score(y_test, dt_preds))
        dt_precision_scores.append(precision_score(y_test, dt_preds))
        dt_accuracy_scores.append(accuracy_score(y_test, dt_preds))

    # # Create an instance of RFE and fit it to the data
    # rfe = RFE(estimator=svm_clf, n_features_to_select=10)
    # rfe.fit(X, y)
    #
    # # Print the selected features
    # print(rfe.support_)

    # Print the mean F1 score, precision, and accuracy for SVM classifier
    print("SVM Classifier:")
    print("F1 Score:", svm_f1_scores)
    print("Precision:", np.mean(svm_precision_scores))
    print("Accuracy:", np.mean(svm_accuracy_scores))

    # Print the mean F1 score, precision, and accuracy for Decision Tree classifier
    print("Decision Tree Classifier:")
    print("F1 Score:", dt_f1_scores)
    print("Precision:", np.mean(dt_precision_scores))
    print("Accuracy:", np.mean(dt_accuracy_scores))

    # save the trained models
    dt_filename = 'Decision_tree_model.pkl'
    with open(dt_filename, 'wb') as file:
        pickle.dump(dt_clf, file)

    svm_filename = 'SVM_model.pkl'
    with open(svm_filename, 'wb') as file:
        pickle.dump(svm_clf, file)






# load data
insulin_df_1 = pd.read_csv("InsulinData.csv", low_memory=False)
cgm_df_1 = pd.read_csv("CGMData.csv", low_memory=False)
insulin_df_2 = pd.read_csv("Insulin_patient2.csv", low_memory=False)
cgm_df_2 = pd.read_csv("CGM_patient2.csv", low_memory=False)

# extract meal data
meal_time_list_1 = mealtime(insulin_df_1)
meal_df_1 = extract_meal_data(cgm_df_1, meal_time_list_1)
meal_time_list_2 = mealtime(insulin_df_2)
meal_df_2 = extract_meal_data(cgm_df_2, meal_time_list_2)
meal_data = pd.concat([meal_df_1, meal_df_2], ignore_index=True)

# extract nomeal data
nomeal_time_list_1 = nomealtime(insulin_df_1)
nomeal_df_1 = extract_no_meal_data(cgm_df_1, nomeal_time_list_1)
nomeal_time_list_2 = nomealtime(insulin_df_2)
nomeal_df_2 = extract_no_meal_data(cgm_df_2, nomeal_time_list_2)
nomeal_data = pd.concat([nomeal_df_1, nomeal_df_2], ignore_index=True)

# extract feature matrix and create training and testing data
meal_part = extract_features(meal_data)
meal_class = np.ones(len(meal_part))
meal_part["class"] = meal_class
meal_part["class"] = meal_part["class"].astype(int)

nomeal_part = extract_features(nomeal_data)
nomeal_class = np.zeros(len(nomeal_part))
nomeal_part["class"] = nomeal_class
nomeal_part["class"] = nomeal_part["class"].astype(int)

data_to_analysis = pd.concat([meal_part, nomeal_part], axis=0, ignore_index=True)
print(data_to_analysis.info())
# train models
model_training(data_to_analysis)



# Glucose Levels in Artificial Pancreas Project Overview:
Data Mining Project:
* Developed a recognition system integrating supervised and unsupervised machine learning techniques for analyzing time series data from two asynchronously operated Medtronic 670G systems collected at 5-minute intervals over a 7-day period.
* Implemented adaptive data cleaning and extracted features, including Fast Fourier Transform (FFT) and Entropy calculations, as well as time span, from the time series dataset.
* Distinguished meal and no meal time series data through the training and testing of machine models, employing sklearn k-fold cross-validation for robust model training.
* Trained Support Vector Machine (SVM) and Decision Tree Machine (DT), achieving a noteworthy DT F1-score of 77% and an Accuracy of 81%. Extracted ground truth and performed clustering using DBSCAN and Kmeans, attaining minimal DBSCAN entropy of 0.22 and maximum DBSCAN purity of 0.83.

## Code and Resources Used 
**Python Version:** 3.11  
**Packages:** pandas, numpy, sklearn, matplotlib, scikit-learn, scipy, pickle  
**Project Source:** Arizona State University, CSE 572 Data Mining

## Dataset:
### Two datasets:
1. From the Continuous Glucose Sensor (CGMData.csv) and
2. from the insulin pump (InsulinData.csv)
### The output of the CGM sensor consists of three columns:
1. Data time stamp (Columns B and C combined),
2. the 5 minute filtered CGM reading in mg/dL, (Column AE) and
3. the ISIG value which is the raw sensor output every 5 mins.
### The output of the pump has the following information:
1. Data time stamp,
2. Basal setting,
3. Micro bolus every 5 mins,
4. Meal intake amount in terms of grams of carbohydrate,
5. Meal bolus,
6. correction bolus,
7. correction factor,
8. CGM calibration or insulin reservoir-related alarms, and
9. auto mode exit events and unique codes representing reasons (Column Q).

## Time Series Extraction
The data is in reverse order of time. This means that the first row is the end of the data collection whereas the last row is the beginning of the data collection. The data starts with manual mode. Manual mode continues until you get a message “AUTO MODE ACTIVE PLGM OFF” in the column “Q” of the InsulinData.csv. From then onwards Auto mode starts. You may get multiple “AUTO MODE ACTIVE PLGM OFF” in column “Q” but only use the earliest one to determine when you switch to auto mode. There is no switching back to manual mode, so the first task is to determine the time stamp when Auto mode starts. The time stamp of the CGM data is not the same as the timestamp of the insulin pump data because these are two different devices which
operate asynchronously.
Once determined the start of Auto Mode from InsulinData.csv, I have to figure out the
timestamp in CGMData.csv where Auto mode starts. This can be done simply by searching for
the time stamp that is nearest to (and later than) the Auto mode start time stamp obtained
from InsulinData.csv. 
For each user, CGM data is first parsed and divided into segments, where each segment corresponds to a day worth of data. One day is considered to start at 12 am and end at 11:59 pm. To compute the percentage with respect to 24 hours, the total number of samples in the
specified range is divided by 288.

### Meal data can be extracted as follows:
From the InsulinData.csv file, search the column Y for a non NAN non zero value. This time
indicates the start of meal consumption time tm. Meal data comprises a 2hr 30 min stretch of
CGM data that starts from tm-30min and extends to tm+2hrs.
### No meal data comprises 2 hrs of raw data that does not have meal intake.

## Data Cleaning
A particular segment may not have all 288 data points. In the data files, those are represented as NaN. To tackle the missing data problem, I tried to do the linear interpolation and directly delete the data from the entire day. According to the metrics I obtained, I choose directly deleting for data cleaning.

## Distinguish Meal and No-meal: Supervised Machine learning Model  

### Extracting features: 
The features were carefully selected and calculated from the oringinal data for model traing, including:
* The climb-up time span
* dG(value change speed from gluce min to max after meal)
* Fast Fourier Transform (FFT)
* Entropy

### Machines and Performance:
*	**Decision Tree Machine** :
   F1-score: 77%
 	Accuracy: 81%
*	**Support Vector Machine**
   F1-score: 65%
 	Accuracy: 73%

## Distinguish Meal and No-meal: Clustering

### Extracting Ground Truth:
Derive the max and min value of meal intake amount from the Y column of the Insulin data. Discretize the meal amount in bins of size 20. Consider each row in the meal data matrix, Put them in the respective bins according to their meal amount label. In total, I should have n = (max-min)/20 bins.

### Performing clustering:
Use the features extracted above to cluster the meal data into n clusters with DBSCAN and KMeans respectively.

### Accuracy of Clustering:
*	**DBSCAN**:
   Entropy: 0.22
 	Purity: 0.83
*	**Kmeans**:
   Entropy: 0.36
 	Purity: 0.0.75

## Acknowledgments
This project was developed as part of the coursework for CSE 572: Data Mining at Arizona State University. Special thanks to the course instructors for their guidance and the provided materials.




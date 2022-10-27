from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings
import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')

#import dataset
dataset = pd.read_csv('diabetes.csv')

# check data head and info
print(dataset.info(), dataset.head())

# histogram
p = dataset.hist(figsize=(20, 20))

corr = dataset.corr()
print(corr)
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)

# treat0
# value 0 untuk pregnan wajar, selainnya harus dicleaning
# replace 0 ke NaN
dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# fungsi untuk get median berdasar outcome


def median_target(var):
    temp = dataset[dataset[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(
        ['Outcome'])[[var]].median().reset_index()
    return temp


# insulin
insulin_median = median_target('Insulin')
# Nan to Mean
dataset.loc[(dataset['Outcome'] == 0) & (
    dataset['Insulin'].isnull()), 'Insulin'] = 102.5
dataset.loc[(dataset['Outcome'] == 1) & (
    dataset['Insulin'].isnull()), 'Insulin'] = 169.5

# glucose
glucose_median = median_target('Glucose')
# Nan to Mean
dataset.loc[(dataset['Outcome'] == 0) & (
    dataset['Glucose'].isnull()), 'Glucose'] = 107
dataset.loc[(dataset['Outcome'] == 1) & (
    dataset['Glucose'].isnull()), 'Glucose'] = 140

# skinthickness
skin_median = median_target('SkinThickness')
# Nan to Mean
dataset.loc[(dataset['Outcome'] == 0) & (
    dataset['SkinThickness'].isnull()), 'SkinThickness'] = 27
dataset.loc[(dataset['Outcome'] == 1) & (
    dataset['SkinThickness'].isnull()), 'SkinThickness'] = 32

# bloodpreasure
blood_median = median_target('BloodPressure')
# Nan to Mean
dataset.loc[(dataset['Outcome'] == 0) & (
    dataset['BloodPressure'].isnull()), 'BloodPressure'] = 70
dataset.loc[(dataset['Outcome'] == 1) & (
    dataset['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

# BMI
bmi_median = median_target('BMI')
# Nan to Mean
dataset.loc[(dataset['Outcome'] == 0) & (
    dataset['BMI'].isnull()), 'BMI'] = 30.1
dataset.loc[(dataset['Outcome'] == 1) & (
    dataset['BMI'].isnull()), 'BMI'] = 34.3

# check sudah gaada null
print(dataset.info())

print(dataset['Glucose'].describe())
print(dataset['Pregnancies'].describe())
print(dataset['BloodPressure'].describe())
print(dataset['SkinThickness'].describe())
print(dataset['Insulin'].describe())
print(dataset['BMI'].describe())
print(dataset['DiabetesPedigreeFunction'].describe())
print(dataset['Age'].describe())

# Data Overview
plt.style.use('ggplot')  # Using ggplot2 style visuals
f, ax = plt.subplots(figsize=(11, 15))
ax.set_facecolor('#fafafa')
ax.set(xlim=(-.05, 200))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data=dataset,
                 orient='h',
                 palette='Set2')

# Before Treating Outliers
# pregnancies plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['Pregnancies'], ax=axes[0], color='red')
axes[0].set_title('Distribution of Pregnancy', fontdict={'fontsize': 8})
axes[0].set_xlabel('Pregnancy Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('Pregnancies', data=dataset,
                     ax=axes[1], orient='v', color='yellow')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# Quantile-based Flooring and Capping metode
print(dataset['Pregnancies'].skew())
print(dataset['Pregnancies'].quantile(0.95))
dataset["Pregnancies"] = np.where(
    dataset["Pregnancies"] > 10, 10, dataset['Pregnancies'])
print(dataset['Pregnancies'].skew())

# After Treating Outliers
# pregnancies plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['Pregnancies'], ax=axes[0], color='red')
axes[0].set_title('Distribution of Pregnancy', fontdict={'fontsize': 8})
axes[0].set_xlabel('Pregnancy Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('Pregnancies', data=dataset,
                     ax=axes[1], orient='v', color='yellow')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# Check Outliers
# glucose plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot02 = sns.distplot(dataset['Glucose'], ax=axes[0], color='red')
axes[0].set_title('Distribution of Glucose', fontdict={'fontsize': 8})
axes[0].set_xlabel('Glucose Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot03 = sns.boxplot('Glucose', data=dataset,
                     ax=axes[1], orient='v', color='yellow')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()
# glucose are acceptable

# Check Outliers
# blood plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['BloodPressure'], ax=axes[0], color='b')
axes[0].set_title('Distribution of BloodPressure', fontdict={'fontsize': 8})
axes[0].set_xlabel('BloodPressure Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()

plot01 = sns.boxplot('BloodPressure', data=dataset,
                     ax=axes[1], orient='v', color='c')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# Quantile-based Flooring and Capping metode
print(dataset['BloodPressure'].skew())
print(dataset['BloodPressure'].quantile(0.95))
print(dataset['BloodPressure'].quantile(0.05))
dataset["BloodPressure"] = np.where(
    dataset["BloodPressure"] > 90, 90, dataset['BloodPressure'])
dataset["BloodPressure"] = np.where(
    dataset["BloodPressure"] < 52, 52, dataset['BloodPressure'])
print(dataset['BloodPressure'].skew())

# Check Outliers
# blood plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['BloodPressure'], ax=axes[0], color='b')
axes[0].set_title('Distribution of BloodPressure', fontdict={'fontsize': 8})
axes[0].set_xlabel('BloodPressure Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()

plot01 = sns.boxplot('BloodPressure', data=dataset,
                     ax=axes[1], orient='v', color='c')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# check outliers
# skin plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['SkinThickness'], ax=axes[0], color='green')
axes[0].set_title('Distribution of SkinThickness', fontdict={'fontsize': 8})
axes[0].set_xlabel('SkinThickness Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('SkinThickness', data=dataset,
                     ax=axes[1], orient='v', color='m')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

print(dataset['SkinThickness'].skew())
print(dataset['SkinThickness'].quantile(0.98))
print(dataset['SkinThickness'].quantile(0.02))
dataset["SkinThickness"] = np.where(
    dataset["SkinThickness"] > 48, 48, dataset['SkinThickness'])
dataset["SkinThickness"] = np.where(
    dataset["SkinThickness"] < 12, 12, dataset['SkinThickness'])

# skin plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['SkinThickness'], ax=axes[0], color='green')
axes[0].set_title('Distribution of SkinThickness', fontdict={'fontsize': 8})
axes[0].set_xlabel('SkinThickness Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('SkinThickness', data=dataset,
                     ax=axes[1], orient='v', color='m')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# BMI plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['BMI'], ax=axes[0], color='m')
axes[0].set_title('Distribution of BMI', fontdict={'fontsize': 8})
axes[0].set_xlabel('BMI Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('BMI', data=dataset, ax=axes[1], orient='v', color='c')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

print(dataset['BMI'].skew())
print(dataset['BMI'].quantile(0.98))
print(dataset['BMI'].quantile(0.02))
dataset["BMI"] = np.where(dataset["BMI"] > 47.5, 47.5, dataset['BMI'])
dataset["BMI"] = np.where(dataset["BMI"] < 20.4, 20.4, dataset['BMI'])

fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['BMI'], ax=axes[0], color='m')
axes[0].set_title('Distribution of BMI', fontdict={'fontsize': 8})
axes[0].set_xlabel('BMI Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('BMI', data=dataset, ax=axes[1], orient='v', color='c')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# dpf plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(
    dataset['DiabetesPedigreeFunction'], ax=axes[0], color='green')
axes[0].set_title('Distribution of DPF', fontdict={'fontsize': 8})
axes[0].set_xlabel('DPF Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('DiabetesPedigreeFunction',
                     data=dataset, ax=axes[1], orient='v')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

print(dataset['DiabetesPedigreeFunction'].skew())
print(dataset['DiabetesPedigreeFunction'].quantile(0.98))
print(dataset['DiabetesPedigreeFunction'].quantile(0.02))
dataset["DiabetesPedigreeFunction"] = np.where(
    dataset["DiabetesPedigreeFunction"] > 1.39, 1.39, dataset['DiabetesPedigreeFunction'])
dataset["DiabetesPedigreeFunction"] = np.where(
    dataset["DiabetesPedigreeFunction"] < 0.12, 0.12, dataset['DiabetesPedigreeFunction'])

# dpf plot
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(
    dataset['DiabetesPedigreeFunction'], ax=axes[0], color='green')
axes[0].set_title('Distribution of DPF', fontdict={'fontsize': 8})
axes[0].set_xlabel('DPF Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('DiabetesPedigreeFunction',
                     data=dataset, ax=axes[1], orient='v')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# age
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['Age'], ax=axes[0], color='green')
axes[0].set_title('Distribution of Age', fontdict={'fontsize': 8})
axes[0].set_xlabel('Age Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('Age', data=dataset, ax=axes[1], orient='v')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

print(dataset['Age'].skew())
print(dataset['Age'].quantile(0.995))
dataset["Age"] = np.where(dataset["Age"] > 69, 69, dataset['Age'])

# age
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8, 4))
plot00 = sns.distplot(dataset['Age'], ax=axes[0], color='green')
axes[0].set_title('Distribution of Age', fontdict={'fontsize': 8})
axes[0].set_xlabel('Age Class', fontdict={'fontsize': 7})
axes[0].set_ylabel('Frequency/Distrubtion', fontdict={'fontsize': 7})
plt.tight_layout()
plot01 = sns.boxplot('Age', data=dataset, ax=axes[1], orient='v')
axes[1].set_title('Five Point Summary', fontdict={'fontsize': 8})
plt.tight_layout()

# data overview
# Data Overview
plt.style.use('ggplot')  # Using ggplot2 style visuals
f, ax = plt.subplots(figsize=(11, 15))
ax.set_facecolor('#fafafa')
ax.set(xlim=(-.05, 200))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data=dataset,
                 orient='h',
                 palette='Set2')

# penasaran apakah insulin seberpengaruh itu?
# 2 datasets
D = dataset[(dataset['Outcome'] != 0)]
H = dataset[(dataset['Outcome'] == 0)]


def plot_distribution(data_select, size_bin):
    # 2 datasets
    tmp1 = D[data_select]
    tmp2 = H[data_select]
    hist_data = [tmp1, tmp2]

    group_labels = ['diabetic', 'healthy']
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                             show_hist=True, bin_size=size_bin, curve_type='kde')

    fig['layout'].update(title=data_select)
    plot(fig)

    #py.iplot(fig, filename = 'Density plot')


plot_distribution('Insulin', 0)
# DROPSS
dataset.drop('Insulin', axis=1, inplace=True)
dataset.drop('SkinThickness', axis=1, inplace=True)
dataset.drop('DiabetesPedigreeFunction', axis=1, inplace=True)

# CLUSTERINGG
X = dataset.iloc[:, [0, 1, 2, 3, 4]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(i)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means to dataset
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
cluster_outcome = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

outcome = dataset.iloc[:, -1].values
arr = [0, 0, 0]
arr_diabetic = [0, 0, 0]
arr_nondiabetic = [0, 0, 0]
arrcount = [0, 0, 0]
labels = kmeans.labels_
for y in range(len(labels)):
    #arr[labels[y]] += outcome[y]
    if outcome[y] == 1:
        arr_diabetic[labels[y]] += 1
    else:
        arr_nondiabetic[labels[y]] += 1
    arrcount[labels[y]] += 1

dataset.drop('Outcome', axis=1, inplace=True)
dataset['Risk'] = cluster_outcome

Xnew = dataset.iloc[:, [0, 1, 2, 3, 4]].values
Y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    Xnew, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit Decision Tree to training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0, 0]+cm[1, 1]+cm[2, 2]) / (cm[0, 0]+cm[0, 1] +
                                           cm[0, 2]+cm[1, 0]+cm[1, 1]+cm[1, 2]+cm[2, 0]+cm[2, 1]+cm[2, 2])

# DIASUMSIKAN DARI CLUSTERING
# 0 -> MEDIUM RISK DIABETIC
# 1 -> HIGH RISK DIABETIC
# 2 -> LOW RISK DIABETIC

pred = classifier.predict([[2, 90, 90, 30.5, 37]])

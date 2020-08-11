# EDA Data Project
# Daidson Alves
# To add a new markdown cell, type '# %% [markdown]'
# %%
# all the imports are done here
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_seq_items = 1000

import warnings
def ignore_warnings(*args, **kwargs):
    pass # ignoring warnings from executions and not from errors
warnings.warn = ignore_warnings

# %% [markdown]
# # 1 - Understanding the data
# * Mostly data exploration;
# * Checking features, their distribution and missing values, dividing them in numerical or categorical.

# %%
#loading the data
data = pd.read_csv("ml_project1_data.csv", sep = ',')

# %% [markdown]
# ### Data characteristics
# We will be checking totals first in order to grasp start working

# %%
#getting to know the shape of the data
print('Shape = (Rows, Columns) ->', data.shape)
print()
data.info()

# %% [markdown]
# ### Looking into the values
# - In order to see what we will be working with, we will check small bits of the data.

# %%
#taking a look at the first rows of the data
data.head()


# %%
#taking a look at the last rows of the data
data.tail()


# %%
#describing the data, getting to know its standard values, quantiles and means
data.describe()

# %% [markdown]
# ### Reviewing types and classifying nulls
# - Main objective is to check if column types are set according to expected (if numeric variables have numeric type and so on). Secondary objective is to get missing values from whole data.

# %%
#printing data types and how many unique values they have
print(data.dtypes)
data.nunique()

# %% [markdown]
# ### Checking null values in the data
# - In order to see if we have good and valid data, we should check how are nulls and missing proportion behaving.

# %%
#cheking null percentage
#data.isnull().sum()
#missing percentage
def missing(df):
    value = df.isna().sum()
    value = value[value>0]
    value_p = value/df.shape[0]
    value_t = value_p>0.05
    return pd.DataFrame({"Total number of missing values" : value, "Missing proportion in Data" : value_p, "Missing >= 5%?" : value_t})

missing(data)


# %%
#dividing the data into numeric, categorical, target and other features
numeric_features = data[["Income", "Kidhome", "Teenhome", "Recency",
                        "MntWines", "MntFruits", "MntMeatProducts",
                        "MntFishProducts", "MntSweetProducts", "MntGoldProds",
                        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
                        "NumStorePurchases", "NumWebVisitsMonth"
                        ]]

categorical_features = data[["Education", "Marital_Status", "AcceptedCmp1",
                            "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
                            "AcceptedCmp5", "Complain"
                            ]]

target = data[["Response"
              ]]

other_features = data[["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"]]

# %% [markdown]
# ### Features disposition so far =
# * 15 numeric features;
# * 8 categorical features;
# * 1 target feature;
# * 5 features amounting other specific data. If needs be, those can be detailed;
# * - "Z_CostContact" and "Z_Revenue" show no significance as they do not vary or have missing types.
# * Our target is the 'Response' feature as we are approaching targeted marketing in order to see how our clientes have responded so far and how we can get them to give us better responses, which means positively answering to our future campaigns.
# * We will add "Dt_Customer" and "Year_Birth" to our numerical features further on as those features can help us predict our target better
# %% [markdown]
# ### Target analysis
# - Understanding the data for distinct target values.

# %%
#count related to client response
data['Response'].value_counts()


# %%
#client response mean
data['Response'].mean()


# %%
#Checking how the data is behaving for those that have had a positive response in the last campaign
response_related_yes = data[data['Response']==1.0]
response_related_yes.describe()


# %%
#Checking how the data is behaving for those that have had a negative response in the last campaign
response_related_no = data[data['Response']==0]
response_related_no.describe()

# %% [markdown]
# # 2 - Customer segmentation
# * Insights on our data so far;
# * Distribution analysis and discrimination.

# %%
#calculating total number of days since the beginning of the relationship
def days_since(dates_series, date_format):
    n = len(dates_series)
    result = [0] * n

    for i in range(n):
        result[i] = (datetime.today()-datetime.strptime(dates_series[i], date_format)).days
    
    return result


# %%
#creating a data frame for all the numeric features, including the new ones proposed above
#(Age and Time since the beginning of the relationship)
num_features = data[["Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
                "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases", 
                "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"]]

def parse_date(td):
    resYear = float(td.days)/364.0                   # get the number of years including the the numbers after the dot
    resMonth = int((resYear - int(resYear))*364/30)  # get the number of months, by multiply the number after the dot by 364 and divide by 30.
    resYear = int(resYear)
    return str(resYear) + "Y" + str(resMonth) + "m"

num_features["DaysRelation"] = days_since(list(other_features.Dt_Customer), "%Y-%m-%d")


# %%
#numeric features distribution so far
num_features.hist(bins=15, figsize=[15,15])
plt.suptitle("Numeric features distribution")
plt.show()


# %%
#correlation between numeric features
corr = num_features.corr()
mask =np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(240, 10, n=9)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .8})

plt.show()

# %% [markdown]
# ### Insights:
# - Most of the data is approaching people close to their late 40's years old;
# - There are slightly more purchases made directly in stores than purchases made using catalogue or through the company's web site;
# - Not many purchases are made in deal mode (NumDealsPurchases). Thus there can be an assumption that discounts are not the main factor for increasing revenue in our targeted campaign;
# - Although we can see that there is a huge amount of visits to the web site, its sales are close to half of the sales made directly in stores;

# %%
#numeric features distribution so far
categorical_features.hist(bins=15, figsize=[15,15], layout=(2, 4))
plt.suptitle("Categorical features distribution")
plt.show()


# %%
#categorical features relation to the "Response" feature
cat_features = ["Education", "Marital_Status", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
                "AcceptedCmp5", "Complain"]

def showdata(vl):
    if vl < .15:
        color = 'red'
    else:
        color = 'black'
    return 'color: %s' % color

# Categorical features analysis
# target = 'Response'
# occurrences threshold = 50 occurrences
# target relation threshold = 15% because of our 'Response' mean that is almost 15%
def cat_feat_describe(df, fc, target, n, thresh):

    fl = []
    if (type(fc)==list):
    
        for feature in fc:
            fl.append(df.groupby([feature]).agg({target : ["count", "mean"]}))    

            fm = pd.concat(fl, keys=fc)

            fm = pd.DataFrame({"Occurrences" : fm.iloc[:,0], "Target relation" : fm.iloc[:,1],
                                 ">= 50 occurrences?" : fm.iloc[:,0]>n})
    else:
        fm = (df.groupby(fc).agg({target : ["count", "mean"]}))
        
        fm = pd.DataFrame({"Number of Occurrences" : fm.iloc[:,0], "Target relation" : fm.iloc[:,1],
                                 ">= 50 occurrences?" : fm.iloc[:,0]>n})
        
    return fm

feat_sum = cat_feat_describe(data, cat_features, "Response", 50, 0.15)
feat_sum.style.applymap(showdata)


# %%
#correlation between categorical features
cat_features = data[["Education", "Marital_Status", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
                "AcceptedCmp5", "Complain"]]
corr = cat_features.corr()
mask =np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(240, 10, n=9)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .8})

plt.show()

# %% [markdown]
# ### Insights:
# - Our threshold for occurrences is of 50, as we are looking at 2% significance of each feature's distinct value;
# - Relation to target is set to 15% because this was our target variable mean value ('Response');
# - There were not many complaints on the campaigns, nonetheless our customers could have felt annoyed by them but they were not, nevertheless they might not be infatuated on our next campaigns.
# - The second campaign was the best one if compared to the other campaigns, even though all the campaigns had a good relation to the 'Response' when we observe the target relation;
# - The only values for 'Marital_Status' that are below 'Response' mean value are Married and Together, showing that these do not tend to have a positive response on our campaigns;
# - PhD and Master values on the 'Education' field are above 'Response' mean value (15%);
# %% [markdown]
# # 3 - Classification model
# * Creating a predictive model using our data;
# * Goal -> maximize profit on our next marketing campaing.

# %%
#detecting anomalies using isolation forest
#creating our train data with all variables
train = data[["Income", "Kidhome", "Teenhome", "Recency",
              "MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds",
              "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
              "NumStorePurchases", "NumWebVisitsMonth", "Education", "Marital_Status", 
              "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain",
              "ID", "Year_Birth", "Response"
              ]]

train["DaysRelation"] = days_since(list(other_features.Dt_Customer), "%Y-%m-%d")


# %%
# linear regression model to input missing values
# preparing our categorical features that have more than 2 distinct values
X = train
y = X["Income"]
y = y[-y.isna()]

X["Marital_Status"] = pd.Categorical(X["Marital_Status"])
X["Marital_Status"] = X["Marital_Status"].cat.codes

X["Education"] = pd.Categorical(X["Education"])
X["Education"] = X["Education"].cat.codes

x_pred = X[X.Income.isna()]
x_pred = x_pred.drop(columns="Income")

X = X[-X.Income.isna()]
X = X.drop(columns="Income")

# Linear Regression Model
reg = LinearRegression().fit(X, y)

# Predictions
y_pred = reg.predict(x_pred)

# Store the predictions in the missing values
train.loc[train.Income.isna(), "Income"] = y_pred


# %%
# checking inconsistencies on year birth feature
print(train[(2020 - train["Year_Birth"]) >=90])


# %%
# since it's not likely that there are people from the 19th century still going on around, we will be cleaning them
X = train
y = X[(2020 - X["Year_Birth"])<=90].Year_Birth
X = X.drop(columns=["ID"])

X["Marital_Status"] = pd.Categorical(X["Marital_Status"])
X["Marital_Status"] = X["Marital_Status"].cat.codes

X["Education"] = pd.Categorical(X["Education"])
X["Education"] = X["Education"].cat.codes

x_pred = X[(2020 - X["Year_Birth"])>=90]
x_pred = x_pred.drop(columns="Year_Birth")

X = X[(2020 - X["Year_Birth"])<90]
X = X.drop(columns="Year_Birth")

# Linear Regression Model
reg = LinearRegression().fit(X, y)

# Predictions
y_pred = reg.predict(x_pred)

# Store the predictions in the missing values
train.loc[(2020 - train["Year_Birth"])>=90, "Year_Birth"] = y_pred.round()
train["Year_Birth"].astype('int')


# %%
#checking if there are any null values in the train data
train.sum().isna()


# %%
#taking a look into the first rows of the train data
train.head()


# %%
#describing the train data, getting its quantiles and means
train.describe()

# %% [markdown]
# ### Analysing outliers using Isolation Forest
# * Score formula = https://miro.medium.com/max/436/1*Zha5PJSauUmig8gstAjflg.png
# * h(x) is the path length of observation x
# * c(n) is the average path length of unsuccessful search in a Binary Search Tree
# * n is the number of external nodes
# * taken from https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e

# %%
#taken from https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
#with steps from https://towardsdatascience.com/feature-engineering-and-data-preparation-using-supermarket-sales-data-part-2-171b7a7a7eb7
def anomaly_plot(df, num_feat_list, l, c):
    sns.set_style("darkgrid") 
    fig, axs = plt.subplots(l, c, figsize=(25, 15), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    for i, column in enumerate(num_feat_list):
        isolation_forest = IsolationForest(n_estimators=1000, contamination="auto")
        isolation_forest.fit(df[column].values.reshape(-1,1))

        xx = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1,1)
        anomaly_score = isolation_forest.decision_function(xx)
        outlier = isolation_forest.predict(xx)
    
        axs[i].plot(xx, anomaly_score, label='Anomaly Score')
        axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                     where=outlier==-1, color='y', 
                     alpha=.4, label='Outlier Region')
        axs[i].legend()
        axs[i].set_title(column)
        
    fig.suptitle('Anomaly Detection', ha='center',
                     va='center', fontsize=20, y=0.92, fontweight='bold')
        
    return

num_feat = ["Income", "Kidhome", "Teenhome", "Recency",
              "MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds",
              "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
              "NumStorePurchases", "NumWebVisitsMonth"]

anomaly_plot(train, num_feat, 5, 3)

# %% [markdown]
# ### Splitting the dataset

# %%
#splitting the dataset using random seeds
#we will use 40% of our base as test and 60% as training
seeds = [3, 27, 91, 150, 475, 2020]
X_train, X_test, y_train, y_test = train_test_split(train, train["Response"], test_size=0.4, random_state=seeds[0])


# %%
#calculating how much was spent on gold products out of the total (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = X_train["MntGoldProds"].iloc[i]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])

X_train["PercentageGoldProducts"] = aux
X_train["PercentageGoldProducts"].head()


# %%
#calculating how much was spent on gold products out of the total (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = X_test["MntGoldProds"].iloc[i]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
    
X_test["PercentageGoldProducts"] = aux
X_test["PercentageGoldProducts"].head()


# %%
#calculating how many campaigns each prospect accepted in the last 5 campaigns (train)
aux = [0]* X_train.shape[0]


for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])
    
    
X_train["TotalAcceptedCampaigns"] = aux
X_train["TotalAcceptedCampaigns"].head()


# %%
#calculating how many campaigns each prospect accepted in the last 5 campaigns (test)
aux = [0]* X_test.shape[0]


for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])
    
    
X_test["TotalAcceptedCampaigns"] = aux
X_test["TotalAcceptedCampaigns"].head()


# %%
#accepted campaigns proportion (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])/5
    
X_train["PercentageAcceptedCampaigns"] = aux
X_train["PercentageAcceptedCampaigns"].head()


# %%
#accepted campaigns proportion (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])/5
    
X_test["PercentageAcceptedCampaigns"] = aux
X_test["PercentageAcceptedCampaigns"].head()


# %%
#calculating how much was spent on wine products out of the total (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntWines"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PercentageWine"] = aux
X_train["PercentageWine"].head()


# %%
#calculating how much was spent on wine products out of the total (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntWines"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PercentageWine"] = aux
X_test["PercentageWine"].head()


# %%
#calculating how much was spent on fruit products out of the total (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntFruits"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PercentageFruit"] = aux
X_train["PercentageFruit"].head()


# %%
#calculating how much was spent on fruit products out of the total (test)
aux = [0]* X_test.shape[0]


for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntFruits"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PercentageFruit"] = aux
X_test["PercentageFruit"].head()


# %%
#calculating how much was spent on meat products out of the total (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntMeatProducts"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PercentageMeat"] = aux
X_train["PercentageMeat"].head()


# %%
#calculating how much was spent on meat products out of the total (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntMeatProducts"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PercentageMeat"] = aux
X_test["PercentageMeat"].head()


# %%
#calculating how much was spent on fish products out of the total (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntFishProducts"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PercentageFish"] = aux
X_train["PercentageFish"].head()


# %%
#calculating how much was spent on fish products out of the total (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntFishProducts"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PercentageFish"] = aux
X_test["PercentageFish"].head()


# %%
#total monetary value spent (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
X_train["ValueSpent"] = aux
X_train["ValueSpent"].head()


# %%
#total monetary value spent (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
X_test["ValueSpent"] = aux
X_test["ValueSpent"].head()


# %%
#consumption potential = amount spent / income (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])/((X_train[["Income"]].iloc[i,:])*2))   
    
X_train["ConPotential"] = aux
X_train["ConPotential"].head()


# %%
#consumption potential = amount spent / income (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])/((X_test[["Income"]].iloc[i,:])*2))
    
X_test["ConPotential"] = aux
X_test["ConPotential"].head()


# %%
#frequency from campaigns (train)
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].iloc[i,:])
    
X_train["Frequency"] = aux
X_train["Frequency"].head()


# %%
#frequency from campaigns (test)
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].iloc[i,:])
    
X_test["Frequency"] = aux
X_test["Frequency"].head()


# %%
#creating a RFM feature - Recency, Frequency and monetary value
#for this we will use our features = Recency, Frequency and ValueSpent
#we will classify our RFM feature according to the quantiles creating a RFM score
#taken from https://www.datacamp.com/community/tutorials/introduction-customer-segmentation-python
#train data
feature_list, n_bins = ["Recency", "Frequency", "ValueSpent"], 5
rfb_dict = {}
for feature in feature_list:
    bindisc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    feature_bin = bindisc.fit_transform(X_train[feature].values[:, np.newaxis])
    feature_bin = pd.Series(feature_bin[:, 0], index=X_train.index)
    feature_bin += 1
    
    if feature == "Recency":
        feature_bin = feature_bin.sub(5).abs() + 1
    rfb_dict[feature+ "_bin"] = feature_bin.astype(int).astype(str)

X_train["RFM"] = (rfb_dict['Recency_bin'] + rfb_dict['Frequency_bin'] + rfb_dict['ValueSpent_bin']).astype(int)
X_train.head()


# %%
#creating a RFM feature - Recency, Frequency and monetary value
#for this we will use our features = Recency, Frequency and ValueSpent
#we will classify our RFM feature according to the quantiles creating a RFM score
#taken from https://www.datacamp.com/community/tutorials/introduction-customer-segmentation-python
#test data
feature_list, n_bins = ["Recency", "Frequency", "ValueSpent"], 5
rfb_dict = {}
for feature in feature_list:
    bindisc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    feature_bin = bindisc.fit_transform(X_test[feature].values[:, np.newaxis])
    feature_bin = pd.Series(feature_bin[:, 0], index=X_test.index)
    feature_bin += 1
    
    if feature == "Recency":
        feature_bin = feature_bin.sub(5).abs() + 1
    rfb_dict[feature+ "_bin"] = feature_bin.astype(int).astype(str)

X_test["RFM"] = (rfb_dict['Recency_bin'] + rfb_dict['Frequency_bin'] + rfb_dict['ValueSpent_bin']).astype(int)
X_test.head()


# %%
#scaling features using MinMaxScaler
num_feat.extend(('PercentageGoldProducts', 'TotalAcceptedCampaigns', 'PercentageAcceptedCampaigns', 'PercentageWine'
                ,'PercentageFruit', 'PercentageMeat', 'PercentageFish', 'ValueSpent'
                ,'ConPotential', 'Frequency', 'RFM'))


# %%
#scikit preprocessing taken from https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
#scaling taken from https://benalexkeen.com/feature-scaling-with-scikit-learn/
#minmax scaling = (x-min(x))/(max(x)-min(x))

suf = '_scaled'
data_scaler = X_train[num_feat]
data_scaler_test = X_test[num_feat]

fscaler = MinMaxScaler()
scaled_d = fscaler.fit_transform(data_scaler.values)
scaled_d_test = fscaler.fit_transform(data_scaler_test.values)

colnames = [s + suf for s in data_scaler.columns]

X_train = pd.concat([X_train, pd.DataFrame(scaled_d, index=data_scaler.index, columns=colnames)], axis=1)
X_test = pd.concat([X_test, pd.DataFrame(scaled_d_test, index=data_scaler_test.index, columns=colnames)], axis=1)


# %%
#taking a look at our processed train data
X_train.head()


# %%
#taking a look at our processed test data
X_test.head()


# %%
#box-cox transformation
#taken from https://www.geeksforgeeks.org/box-cox-transformation-using-python/
#receives a dataframe consisting only of scaled features and the target, and the name of the target feature.
#returns both the dataframe with the features already transformed to the best transformation and a dictionary
#with the name of each feature with its best transformation name.
def power_transf(df, target_feat):

    # define a set of transformations
    trans_dict = {"x": lambda x: x, "log": np.log, "sqrt": np.sqrt, 
                  "exp": np.exp, "**1/4": lambda x: np.power(x, 0.25), 
                  "**2": lambda x: np.power(x, 2), "**4": lambda x: np.power(x, 4)}

    target = target_feat
    best_power_dict = {}
    for feature in df.columns[:-1]:
        max_test_value, max_trans, best_power_trans = 0, "", None
        for trans_key, trans_value in trans_dict.items():
            # apply transformation
            feature_trans = trans_value(df[feature])
            if trans_key == "log":
                feature_trans.loc[np.isfinite(feature_trans)==False] = -50.

            # bin feature
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            feature_bin = bindisc.fit_transform(feature_trans.values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=df.index)

            # obtain contingency table
            df_ = pd.DataFrame(data={feature: feature_bin, target: df[target]})
            cont_tab = pd.crosstab(df_[feature], df_[target], margins = False)        

            # compute Chi-Squared
            chi_test_value = stats.chi2_contingency(cont_tab)[0]
            if chi_test_value > max_test_value:
                max_test_value, max_trans, best_power_trans = chi_test_value, trans_key, feature_trans      

        best_power_dict[feature] = (max_test_value, max_trans, best_power_trans)
        df[feature] = best_power_trans
        
    return df, best_power_dict


# %%
def power_transf(X_train, X_test, target_feat):

    # define a set of transformations
    trans_dict = {"x": lambda x: x, "log": np.log, "sqrt": np.sqrt, 
                  "exp": np.exp, "**1/4": lambda x: np.power(x, 0.25), 
                  "**2": lambda x: np.power(x, 2), "**4": lambda x: np.power(x, 4)}

    target = target_feat
    best_power_dict = {}
    for feature in X_train.columns[:-1]:
        max_test_value, max_trans, best_power_trans = 0, "", None
        for trans_key, trans_value in trans_dict.items():
            # apply transformation
            feature_trans = trans_value(X_train[feature])
            feature_trans_test = trans_value(X_test[feature])
            if trans_key == "log":
                feature_trans.loc[np.isfinite(feature_trans)==False] = -50.
                feature_trans_test.loc[np.isfinite(feature_trans_test)==False] = -50.

            # bin feature
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            feature_bin = bindisc.fit_transform(feature_trans.values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=X_train.index)

            # obtain contingency table
            df_ = pd.DataFrame(data={feature: feature_bin, target: X_train[target]})
            cont_tab = pd.crosstab(df_[feature], df_[target], margins = False)        

            # compute Chi-Squared
            chi_test_value = stats.chi2_contingency(cont_tab)[0]
            if chi_test_value > max_test_value:
                max_test_value, max_trans, best_power_trans = chi_test_value, trans_key, feature_trans      

        best_power_dict[feature] = (max_test_value, max_trans, best_power_trans)
        X_train[feature] = best_power_trans
        
    return X_train, X_test, best_power_dict


# %%
#we need to create a dataframe containing only the scaled features with the Response.
df_pt = X_train.iloc[:,-15:]
df_pt_test = X_test.iloc[:,-15:]

df_pt["Response"] = X_train["Response"]

data_pt, data_pt_test, best_pt = power_transf(df_pt, df_pt_test, "Response")

print("Best Power Transformation for each feature:")
for key in best_pt:
    print("\t->", key, best_pt[key][1])


# %%
#replacing the old columns of scaled features with the features transformed according to the best transformation
coln = data_pt.columns[:-1]

X_train.drop(columns=coln, inplace=True)
X_train[coln] = data_pt[coln]

X_test.drop(columns=coln, inplace=True)
X_test[coln] = data_pt_test[coln]


# %%
X_train.head()


# %%
X_test.head()


# %%
#merging categories
#in Marital_Status:  "Single" as 3, "Widow" as 2, "Divorced" as 1 and ["Married", "Together"] as 0
X_train["Marital_Status_bin"] = X_train['Marital_Status'].apply(lambda x: 3 if x == "Single" else
                                                            (2 if x == "Widow" else
                                                             (1 if x == "Divorced" else 0))).astype(int)

X_test["Marital_Status_bin"] = X_test['Marital_Status'].apply(lambda x: 3 if x == "Single" else
                                                            (2 if x == "Widow" else
                                                             (1 if x == "Divorced" else 0))).astype(int)


# %%
#in Education: "Phd" as 2, "Master" as 1 and ['Graduation', 'Basic', '2n Cycle'] as 0
X_train["Education_bin"] = X_train['Education'].apply(lambda x: 2 if x == "PhD" else (1 if x == "Master" else 0)).astype(int)

X_test["Education_bin"] = X_test['Education'].apply(lambda x: 2 if x == "PhD" else (1 if x == "Master" else 0)).astype(int)


# %%
#converting Kidhome and Teenhome to int
X_train["Kidhome"] = X_train['Kidhome'].astype(int)
X_train["Teenhome"] = X_train['Teenhome'].astype(int)

X_test["Kidhome"] = X_test['Kidhome'].astype(int)
X_test["Teenhome"] = X_test['Teenhome'].astype(int)


# %%
#creating a new feature called HasOffspring, which defines if the client has children, be a kid or a teenager, or not
#train data
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    if(int(X_train[["Kidhome"]].iloc[i,:])+int(X_train[["Teenhome"]].iloc[i,:])>0):
        aux[i] = 1
    else:
        aux[i] = 0
    
X_train["HasOffspring"] = aux
X_train["HasOffspring"].head()


# %%
#creating a new feature called HasOffspring, which defines if the client has children, be a kid or a teenager, or not
#test data
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    if(int(X_test[["Kidhome"]].iloc[i,:])+int(X_test[["Teenhome"]].iloc[i,:])>0):
        aux[i] = 1
    else:
        aux[i] = 0
    
X_test["HasOffspring"] = aux
X_test["HasOffspring"].head()


# %%
#look into our train data
X_train.head()


# %%
#look into our test data
X_test.head()


# %%
#principal component analysis - PCA
#we will drop most of our features

X_train.drop(['Education','Marital_Status','Year_Birth','Year_Birth',
            'Income', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'DaysRelation', 'PercentageGoldProducts' ,'TotalAcceptedCampaigns',
            'PercentageAcceptedCampaigns', 'PercentageWine', 'PercentageFruit',
            'PercentageMeat', 'PercentageFish', 'ValueSpent', 'ConPotential',
            'Frequency', 'RFM'], 
           axis=1, inplace=True)

X_test.drop(['Education','Marital_Status','Year_Birth','Year_Birth',
            'Income', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'DaysRelation', 'PercentageGoldProducts' ,'TotalAcceptedCampaigns',
            'PercentageAcceptedCampaigns', 'PercentageWine', 'PercentageFruit',
            'PercentageMeat', 'PercentageFish', 'ValueSpent', 'ConPotential',
            'Frequency', 'RFM'], 
           axis=1, inplace=True)


# %%
#final adjustments for our PC analysis
columns = X_train.columns
columns = columns.drop(['Kidhome', 'Teenhome','AcceptedCmp3', 'AcceptedCmp4',
                        'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response',
                        'Marital_Status_bin', 'Education_bin', 'HasOffspring'])

pca = PCA(n_components=2)
pcomponents = pca.fit_transform(X_train[columns])
pcomponents_test = pca.fit_transform(X_test[columns])

X_train['pca1'] = pcomponents[:,0]
X_train['pca2'] = pcomponents[:,1]
X_test['pca1'] = pcomponents_test[:,0]
X_test['pca2'] = pcomponents_test[:,1]


# %%
#cheking PCA feature values created on train data
X_train.head()


# %%
#cheking PCA feature values created on test data
X_test.head()


# %%
#rearranging a list of numeric features
num_feat = ['Income_scaled', 'Recency_scaled', 'MntWines_scaled', 'MntFruits_scaled',
       'MntMeatProducts_scaled', 'MntFishProducts_scaled', 'MntSweetProducts_scaled',
       'MntGoldProds_scaled', 'NumDealsPurchases_scaled', 'NumWebPurchases_scaled',
       'NumCatalogPurchases_scaled', 'NumStorePurchases_scaled', 'NumWebVisitsMonth_scaled',
       'PercentageGoldProducts_scaled', 'TotalAcceptedCampaigns_scaled', 'PercentageAcceptedCampaigns_scaled',
       'PercentageWine_scaled', 'PercentageFruit_scaled', 'PercentageMeat_scaled', 'PercentageFish_scaled', 'ValueSpent_scaled',
       'ConPotential_scaled', 'Frequency_scaled', 'RFM_scaled', 'pca1', 'pca2']


# %%
# getting dummies for categorical features
cat_columns = ['Kidhome', 'Teenhome','Marital_Status_bin','Education_bin']
X_train = pd.get_dummies(X_train, prefix_sep="_",
                              columns=cat_columns)
X_test = pd.get_dummies(X_test, prefix_sep="_",
                              columns=cat_columns)


# %%
#train data with the dummies
X_train.head()


# %%
#test data with the dummies
X_test.head()


# %%
#rearranging a list of categorical features
cat_feat = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4",
         "AcceptedCmp5", "Complain", "HasOffspring", ]
cat_feat.extend((['Kidhome_0', 'Kidhome_1', 'Kidhome_2', 'Teenhome_0', 'Teenhome_1', 'Teenhome_2',
                  'Marital_Status_bin_0', 'Education_bin_0']))


# %%
#feature selection
#inputs = dataframe, a list of continuous features names, a list of categorical features names,
#the name of the target feature and returns a dataframe with the discrimination ability of each feature and if
#its p-value is lower than 0.05.
#10 is the default number of bins and uniform is the strategy used in the binning of continuous features.
#taken from https://www.kaggle.com/rodsaldanha/data-analysis-campaign
def chisq_ranker(df, continuous_flist, categorical_flist, target, n_bins=10, binning_strategy="uniform"):
    chisq_dict = {}
    if  continuous_flist:
        bindisc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', 
                               strategy=binning_strategy)
        for feature in continuous_flist:            
            feature_bin = bindisc.fit_transform(df[feature].values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=df.index)
            cont_tab = pd.crosstab(feature_bin, df[target], margins = False)
            chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2] 
    if  categorical_flist:
        for feature in categorical_flist:  
            cont_tab = pd.crosstab(df[feature], df[target], margins = False)          
            chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]       
    
    df_chi = pd.DataFrame(chisq_dict, index=["Chi-Squared", "p-value"]).transpose()
    df_chi.sort_values("Chi-Squared", ascending=False, inplace=True)
    df_chi["valid"]=df_chi["p-value"]<=0.05
    
    
    return df_chi


# %%
#using our chisq ranker
df_chisq_rank = chisq_ranker(X_train, num_feat, cat_feat, "Response")
df_chisq_rank.head(15)


# %%
#analysing the features worth (or value) in the data
sns.set_style('whitegrid') 

plt.subplots(figsize=(13,12))
pal = sns.color_palette("RdBu_r", len(df_chisq_rank))
rank = df_chisq_rank['Chi-Squared'].argsort().argsort()  

sns.barplot(y=df_chisq_rank.index,x=df_chisq_rank['Chi-Squared'], palette=np.array(pal[::-1])[rank])
plt.title("Features worth by Chi-Squared statistic test", fontsize=18)
plt.ylabel("Input feature", fontsize=14)
plt.xlabel("Chi-square", fontsize=14)

plt.show()


# %%
#balance training set
y_train.value_counts().plot(kind='bar', title='Count (target)');


# %%
#predictive model
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values))
X_test = scaler.transform(X_test)


# %%
#logistic regression
LR = LogisticRegression()
LR.fit(X_train, y_train)


# %%
#calculating ROC score from the predict
y_pred = LR.predict(X_test)
print('ROC score: {}'.format(roc_auc_score(y_test, y_pred)))


# %%
#setting the Gaussian Naive-Bayes classification
NB = GaussianNB()
NB.fit(X_train, y_train)


# %%
#calculating ROC score after the Gaussian-Bayes classification
y_pred = LR.predict(X_test)
print('ROC score: {}'.format(roc_auc_score(y_test, y_pred)))


# %%
#using Multilayer Perceptron (MLP) to classify our model
#it receives 'n' inputs and each of them is weighted
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000)
mlp.fit(X_train, y_train.values.ravel())


# %%
#testing the data 
predictions = mlp.predict(X_test)


# %%
#printing the predictions array
print(predictions)


# %%
#Last thing: evaluation of algorithm performance in classifying flowers
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))


# %%
#printing the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# %%
#checking fit for the last ROC score
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
print('ROC score: {}'.format(roc_auc_score(y_test, y_pred)))

# %% [markdown]
# # 4 - Business presentation
# * It was a good database to work with because there were not many missing values or confusing data;
# * In order to achieve the best results possible, new features were created. With this we could see that those who have already responded to previous campaigns and those who have spent the most so far are more likely to give us positive responses in the next campaigns;
# * The consumer elimination was minimum in order to preserve original data to the most. One of the next evaluations that can be done is eliminating people that have not responded to any of our campaigns;
# * Analysing recency, frequency, total value spent and number of accepted campaigns was a way of engineering new features that could aggregate significant value to the work;
# * The proportion of money spent in each product can help us directing and targeting those who have responded to previous campaigns with specific products for their likes and buys;
# * When observing frequency, we could not assume whereas our consumers would tend to buy more according to it;
# * Having kids and/or teens at home made no significant difference on positively answering the campaigns;
# * Complain was not observed to be related to not answering the campaings. As a matter of fact, it did not change much our feature analysis;
# * Meat and Wine were the most significant features so a new targeted campaign giving deals and discounts on their prices can get us better outcomes;
# * Nonetheless discounts were given on deal products, this was not a major feature on our data as people tended to answer the campaigns no matter there was a discount.

# %%

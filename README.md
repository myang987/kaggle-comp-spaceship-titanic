# Introduction
This notebook is a personal exploration of an end-to-end data science process and serves as a learning process for myself as I continue to pursue a career in data science. <br>

I am completing a submission for the Kaggle Spaceship-Titanic Competition. The goal of the competition is to create a machine learning model that can predict passenger outcome after a spaceship crash scenario. 

_"The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage...the unwary Spaceship Titanic collided with a spacetime anomaly...To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system._

#### **Evaluation Metric**

Submissions are evaluated based on their classification accuracy, the percentage of predicted labels that are correct.




### Imports


```python
# Core
import pandas as pd
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Reading Data Inputs


```python
# !unzip data/spaceship-titanic.zip -d data/

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
pd.set_option("display.max_rows", None)
```

# 1. Exploratory Data Analysis

## 1.1 Preliminary Observations


```python
print("train: ", train.shape)
train.head(5)
```

    train:  (8693, 14)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("test: ", test.shape)
test.head(5)
```

    test:  (4277, 13)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0013_01</td>
      <td>Earth</td>
      <td>True</td>
      <td>G/3/S</td>
      <td>TRAPPIST-1e</td>
      <td>27.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Nelly Carsoning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0018_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/4/S</td>
      <td>TRAPPIST-1e</td>
      <td>19.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2823.0</td>
      <td>0.0</td>
      <td>Lerome Peckers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0019_01</td>
      <td>Europa</td>
      <td>True</td>
      <td>C/0/S</td>
      <td>55 Cancri e</td>
      <td>31.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Sabih Unhearfus</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0021_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>C/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>38.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>6652.0</td>
      <td>0.0</td>
      <td>181.0</td>
      <td>585.0</td>
      <td>Meratz Caltilter</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0023_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/5/S</td>
      <td>TRAPPIST-1e</td>
      <td>20.0</td>
      <td>False</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>635.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Brence Harperez</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8693 entries, 0 to 8692
    Data columns (total 14 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   PassengerId   8693 non-null   object 
     1   HomePlanet    8492 non-null   object 
     2   CryoSleep     8476 non-null   object 
     3   Cabin         8494 non-null   object 
     4   Destination   8511 non-null   object 
     5   Age           8514 non-null   float64
     6   VIP           8490 non-null   object 
     7   RoomService   8512 non-null   float64
     8   FoodCourt     8510 non-null   float64
     9   ShoppingMall  8485 non-null   float64
     10  Spa           8510 non-null   float64
     11  VRDeck        8505 non-null   float64
     12  Name          8493 non-null   object 
     13  Transported   8693 non-null   bool   
    dtypes: bool(1), float64(6), object(7)
    memory usage: 891.5+ KB



```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4277 entries, 0 to 4276
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   PassengerId   4277 non-null   object 
     1   HomePlanet    4190 non-null   object 
     2   CryoSleep     4184 non-null   object 
     3   Cabin         4177 non-null   object 
     4   Destination   4185 non-null   object 
     5   Age           4186 non-null   float64
     6   VIP           4184 non-null   object 
     7   RoomService   4195 non-null   float64
     8   FoodCourt     4171 non-null   float64
     9   ShoppingMall  4179 non-null   float64
     10  Spa           4176 non-null   float64
     11  VRDeck        4197 non-null   float64
     12  Name          4183 non-null   object 
    dtypes: float64(6), object(7)
    memory usage: 434.5+ KB



```python
# Percentages of missing values
train.isna().mean().sort_values(ascending=False)
```




    CryoSleep       0.024963
    ShoppingMall    0.023927
    VIP             0.023352
    HomePlanet      0.023122
    Name            0.023007
    Cabin           0.022892
    VRDeck          0.021627
    FoodCourt       0.021051
    Spa             0.021051
    Destination     0.020936
    RoomService     0.020821
    Age             0.020591
    PassengerId     0.000000
    Transported     0.000000
    dtype: float64



### Numerical Columns from train


```python
# List of numerical features
num_features = train.select_dtypes(exclude='object').copy()
print("Number of numerical features: ", len(num_features.columns))
num_features.columns
```

    Number of numerical features:  7





    Index(['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
           'Transported'],
          dtype='object')




```python
num_features.describe().round(decimals=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8514.00</td>
      <td>8512.00</td>
      <td>8510.00</td>
      <td>8485.00</td>
      <td>8510.00</td>
      <td>8505.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.83</td>
      <td>224.69</td>
      <td>458.08</td>
      <td>173.73</td>
      <td>311.14</td>
      <td>304.85</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.49</td>
      <td>666.72</td>
      <td>1611.49</td>
      <td>604.70</td>
      <td>1136.71</td>
      <td>1145.72</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.00</td>
      <td>47.00</td>
      <td>76.00</td>
      <td>27.00</td>
      <td>59.00</td>
      <td>46.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79.00</td>
      <td>14327.00</td>
      <td>29813.00</td>
      <td>23492.00</td>
      <td>22408.00</td>
      <td>24133.00</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical Features from train


```python
cat_features = train.select_dtypes(include='object').copy()
print("Number of categorical features: ", len(cat_features.columns))
cat_features.columns
```

    Number of categorical features:  7





    Index(['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP',
           'Name'],
          dtype='object')




```python
cat_features.describe().round(decimals=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>VIP</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8693</td>
      <td>8492</td>
      <td>8476</td>
      <td>8494</td>
      <td>8511</td>
      <td>8490</td>
      <td>8493</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>8693</td>
      <td>3</td>
      <td>2</td>
      <td>6560</td>
      <td>3</td>
      <td>2</td>
      <td>8473</td>
    </tr>
    <tr>
      <th>top</th>
      <td>0001_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>G/734/S</td>
      <td>TRAPPIST-1e</td>
      <td>False</td>
      <td>Gollux Reedall</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>4602</td>
      <td>5439</td>
      <td>8</td>
      <td>5915</td>
      <td>8291</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Notes for Data Cleaning & Processing:

None of the features have significant amounts of missing values. The highest being "CryoSleep" with 2.5% of values missing. There is potential to predict missing values based on other features using an algorithm such as KNN Imputation.

## 1.2 Univariate Analysis

First, it is good practice to evaluate the skew of the target column as it may adversely affect the outcome of the prediction accuracy of regression models. This is not required (or possible) for our dataset as the target is a binary variable.

Note: Correcting skew is important for Linear Regression, but not necessary for Decision Trees and Random Forests.


```python
plt.figure()
sns.histplot(
    train.Transported, kde=True,
    stat="percent", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4)
)
plt.title('Distribution of Transported')
plt.show()
```


    
![png](README_files/README_20_0.png)
    


### Numerical Features


```python
fig = plt.figure(figsize=(12,18))
for i in range(len(num_features.columns)):
    fig.add_subplot(9,4,i+1)
    sns.histplot(
    num_features.iloc[:,i].dropna(), kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4)
)
    plt.xlabel(num_features.columns[i])

plt.tight_layout(pad=1.0)
```


    
![png](README_files/README_22_0.png)
    


The numerical features, besides "Age" and "Transported", all are heavily right skewed. Based on the data field descriptions, "RoomService", "FoodCourt", "ShoppingMall", "Spa", and "VRDeck" all indicate the total amount of money billed to each passenger for each service throughout the duration of the voyage. The heavy right skew makes sense as the distribution for money spent on a luxury.


```python
fig = plt.figure(figsize=(12, 18))

for i in range(len(num_features.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=num_features.iloc[:,i])

plt.tight_layout()
```


    
![png](README_files/README_24_0.png)
    


Taking a look at potential outliers, there only exists a few points within RoomService and ShoppingMall that seem to be outliers.


```python
train.sort_values(by='ShoppingMall', ascending=False).head(10)[['ShoppingMall']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ShoppingMall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8415</th>
      <td>23492.0</td>
    </tr>
    <tr>
      <th>6223</th>
      <td>12253.0</td>
    </tr>
    <tr>
      <th>7425</th>
      <td>10705.0</td>
    </tr>
    <tr>
      <th>4724</th>
      <td>10424.0</td>
    </tr>
    <tr>
      <th>5673</th>
      <td>9058.0</td>
    </tr>
    <tr>
      <th>6453</th>
      <td>7810.0</td>
    </tr>
    <tr>
      <th>385</th>
      <td>7185.0</td>
    </tr>
    <tr>
      <th>8315</th>
      <td>7148.0</td>
    </tr>
    <tr>
      <th>637</th>
      <td>7104.0</td>
    </tr>
    <tr>
      <th>5137</th>
      <td>6805.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.sort_values(by='RoomService', ascending=False).head(10)[['RoomService']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RoomService</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4416</th>
      <td>14327.0</td>
    </tr>
    <tr>
      <th>5105</th>
      <td>9920.0</td>
    </tr>
    <tr>
      <th>8626</th>
      <td>8586.0</td>
    </tr>
    <tr>
      <th>7933</th>
      <td>8243.0</td>
    </tr>
    <tr>
      <th>7118</th>
      <td>8209.0</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>8168.0</td>
    </tr>
    <tr>
      <th>4762</th>
      <td>8151.0</td>
    </tr>
    <tr>
      <th>5725</th>
      <td>8142.0</td>
    </tr>
    <tr>
      <th>928</th>
      <td>8030.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>7406.0</td>
    </tr>
  </tbody>
</table>
</div>



Notes for Data Cleaning & Processing:

I can log transform the right skewed features to get a more normal distribution. This would reduce the impact of the outliers while eliminating the need to remove observation points from the training dataset. Given that there is still a lot of data points on the tails of the distribution, eliminating too many points may reduce accuracy.

### Categorical Features


```python
cat_features.columns
```




    Index(['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP',
           'Name'],
          dtype='object')




```python
cat_features_visual = cat_features[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]

fig = plt.figure(figsize=(18,20))
for index in range(len(cat_features_visual.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat_features_visual.iloc[:,index], data=cat_features_visual.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
```


    
![png](README_files/README_31_0.png)
    


There doesn't seem to be anything out of the ordinary for the non-identifier categorical variables.


```python
train.Cabin.head(5)
```




    0    B/0/P
    1    F/0/S
    2    A/0/S
    3    A/0/S
    4    F/1/S
    Name: Cabin, dtype: object



Notes for Feature Engineering:

One interesting thing to note relates to the "Cabin" feature. The data in this feature indicates deck/num/side (where side indicates P - Port, S - Starboard). I can create additional features that extract the location of a passenger's cabin. This information could prove useful given location of cabins typically highly affect survival outcomes in disaster scenarios.

## Bivariate Analysis


```python
plt.figure(figsize=(7,6))
plt.title('Correlation of numerical attributes', size=12)
correlation = num_features.corr()
sns.heatmap(correlation)
```




    <Axes: title={'center': 'Correlation of numerical attributes'}>




    
![png](README_files/README_36_1.png)
    



```python
correlation = train.select_dtypes(exclude=['object']).corr()
correlation[['Transported']].sort_values(['Transported'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Transported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Transported</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>FoodCourt</th>
      <td>0.046566</td>
    </tr>
    <tr>
      <th>ShoppingMall</th>
      <td>0.010141</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.075026</td>
    </tr>
    <tr>
      <th>VRDeck</th>
      <td>-0.207075</td>
    </tr>
    <tr>
      <th>Spa</th>
      <td>-0.221131</td>
    </tr>
    <tr>
      <th>RoomService</th>
      <td>-0.244611</td>
    </tr>
  </tbody>
</table>
</div>



At first glance, RoomService, Spa, and VRDeck are more inversely related to the target Transport. This may be because of the location of these services were more impacted during the disaster incident. Higher spenders were more likely to be in these locations during the time of the incident.

There is some correlation between the other features themselves, but not significant enough to invoke multicollinearity skewing.


```python
sns.pairplot(train, hue="Transported")
```




    <seaborn.axisgrid.PairGrid at 0x10a8fd400>




    
![png](README_files/README_39_1.png)
    


Looking at the pairwise plots, my main hypothesis is that location aboard the ship is the main factor in whether a passenger was transported at the time of the incident. This more enforces my idea of feature engineering the cabin locations for training.

# 2. Data Cleaning

Steps:
1. Removing redundant features
2. Filling missing data
3. Dealing with outliers

Since I will be log-transforming feature values, I need to populate missing values first before removing outliers. In the case that I'd be removing outliers, intorducing artificial data could potentially skew and alter what is considered an outlier. But since I am log transforming with a predicted value, this risk should be minimized.

## 2.1 Removing redundant features

There aren't any features that I feel need to be removed due to multicollinearity.

## 2.2 Filling in Missing Values

### Cabin
This feature is the most important to get accurate when filling missing data because I will be conducting feature engineering on the values. Because of this, and the fact that only 2% of values are missing, I will remove rows where this is missing. I do not want to populate with artifical values that will then have propogating effects through feature engineering without knowing the importance/correlation to the target variable.

### Name

Since identifier column, populate with "No Name". 

### Age

Populate with Median age.

### VIP

Populate with 0, assume no VIP status.

### HomePlanet, Destination
Populate with most common locations.

### ShoppingMall, FoodCourt, RoomService, Spa, VRDeck

Set 0 as default for any missing values. Assuming no purchases.

### Cryosleep

CryoSleep is defined as a passenger sleeping for the whole duration of the trip. I can populate the missing values based on if any other feature has value. Ie. If the passenger spent money at the mall, they can't be in cryosleep.






```python
train_clean = train.copy()

## Cabin
train_clean = train_clean.loc[~train_clean['Cabin'].isnull()]

## Name
train_clean['Name'].fillna('No Name', inplace=True)

## Age
train_clean['Age'].fillna(train_clean['Age'].median(), inplace=True)

## VIP
train_clean['VIP'].fillna(0, inplace=True)

## HomePlanet, Destination
train_clean['HomePlanet'].fillna(train_clean['HomePlanet'].mode()[0], inplace=True)
train_clean['Destination'].fillna(train_clean['Destination'].mode()[0], inplace=True)

## ShoppingMall, FoodCourt, RoomService, Spa, VRDeck
check_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in check_cols:
    train_clean[col].fillna(0, inplace=True)

## CryoSleep
mask = train_clean['CryoSleep'].isna()

train_clean.loc[mask, 'CryoSleep'] = np.where(
    (train_clean.loc[mask, check_cols] > 0).any(axis=1), 
    0,  # if any > 0
    1   # else
)

```

## 2.3 Dealing with Outliers

I will log transform the heavily right skewed numerical features.


```python
features_to_transform = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

fig = plt.figure(figsize=(12, 18))

for i, col in enumerate(features_to_transform):
    if col in train.columns:
        fig.add_subplot(9, 4, i + 1)
        sns.histplot(
            train[col].dropna(), kde=True,
            stat="density", kde_kws=dict(cut=3),
            alpha=.4, edgecolor=(1, 1, 1, .4)
        )
        plt.xlabel(col)

plt.tight_layout(pad=1.0)
```


    
![png](README_files/README_48_0.png)
    



```python
for feature in features_to_transform:
    train_clean[f'{feature}_log'] = np.log1p(train_clean[feature])
    test[f'{feature}_log'] = np.log1p(test[feature])
```


```python
tmp = ['RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log']

fig = plt.figure(figsize=(12, 18))

for i, col in enumerate(tmp):
    if col in train_clean.columns:
        fig.add_subplot(9, 4, i + 1)
        sns.histplot(
            train_clean[col].dropna(), kde=True,
            stat="density", kde_kws=dict(cut=3),
            alpha=.4, edgecolor=(1, 1, 1, .4)
        )
        plt.xlabel(col)

plt.tight_layout(pad=1.0)
```


    
![png](README_files/README_50_0.png)
    



```python
train_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8494 entries, 0 to 8692
    Data columns (total 19 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   PassengerId       8494 non-null   object 
     1   HomePlanet        8494 non-null   object 
     2   CryoSleep         8494 non-null   object 
     3   Cabin             8494 non-null   object 
     4   Destination       8494 non-null   object 
     5   Age               8494 non-null   float64
     6   VIP               8494 non-null   object 
     7   RoomService       8494 non-null   float64
     8   FoodCourt         8494 non-null   float64
     9   ShoppingMall      8494 non-null   float64
     10  Spa               8494 non-null   float64
     11  VRDeck            8494 non-null   float64
     12  Name              8494 non-null   object 
     13  Transported       8494 non-null   bool   
     14  RoomService_log   8494 non-null   float64
     15  FoodCourt_log     8494 non-null   float64
     16  ShoppingMall_log  8494 non-null   float64
     17  Spa_log           8494 non-null   float64
     18  VRDeck_log        8494 non-null   float64
    dtypes: bool(1), float64(11), object(7)
    memory usage: 1.2+ MB


# 3. Feature Engineering


```python
train_clean[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = train_clean['Cabin'].str.split('/', expand=True)
test[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = test['Cabin'].str.split('/', expand=True)
```


```python
train_final = pd.get_dummies(train_clean[['Transported', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log', 'Cabin_Deck', 'Cabin_Side']])
train_final.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8494 entries, 0 to 8692
    Data columns (total 27 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   Transported                8494 non-null   bool   
     1   Age                        8494 non-null   float64
     2   RoomService_log            8494 non-null   float64
     3   FoodCourt_log              8494 non-null   float64
     4   ShoppingMall_log           8494 non-null   float64
     5   Spa_log                    8494 non-null   float64
     6   VRDeck_log                 8494 non-null   float64
     7   HomePlanet_Earth           8494 non-null   bool   
     8   HomePlanet_Europa          8494 non-null   bool   
     9   HomePlanet_Mars            8494 non-null   bool   
     10  CryoSleep_False            8494 non-null   bool   
     11  CryoSleep_True             8494 non-null   bool   
     12  Destination_55 Cancri e    8494 non-null   bool   
     13  Destination_PSO J318.5-22  8494 non-null   bool   
     14  Destination_TRAPPIST-1e    8494 non-null   bool   
     15  VIP_False                  8494 non-null   bool   
     16  VIP_True                   8494 non-null   bool   
     17  Cabin_Deck_A               8494 non-null   bool   
     18  Cabin_Deck_B               8494 non-null   bool   
     19  Cabin_Deck_C               8494 non-null   bool   
     20  Cabin_Deck_D               8494 non-null   bool   
     21  Cabin_Deck_E               8494 non-null   bool   
     22  Cabin_Deck_F               8494 non-null   bool   
     23  Cabin_Deck_G               8494 non-null   bool   
     24  Cabin_Deck_T               8494 non-null   bool   
     25  Cabin_Side_P               8494 non-null   bool   
     26  Cabin_Side_S               8494 non-null   bool   
    dtypes: bool(21), float64(6)
    memory usage: 638.7 KB



```python
from sklearn.ensemble import RandomForestClassifier
```


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4277 entries, 0 to 4276
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   PassengerId       4277 non-null   object 
     1   HomePlanet        4190 non-null   object 
     2   CryoSleep         4184 non-null   object 
     3   Cabin             4177 non-null   object 
     4   Destination       4185 non-null   object 
     5   Age               4186 non-null   float64
     6   VIP               4184 non-null   object 
     7   RoomService       4195 non-null   float64
     8   FoodCourt         4171 non-null   float64
     9   ShoppingMall      4179 non-null   float64
     10  Spa               4176 non-null   float64
     11  VRDeck            4197 non-null   float64
     12  Name              4183 non-null   object 
     13  RoomService_log   4195 non-null   float64
     14  FoodCourt_log     4171 non-null   float64
     15  ShoppingMall_log  4179 non-null   float64
     16  Spa_log           4176 non-null   float64
     17  VRDeck_log        4197 non-null   float64
     18  Cabin_Deck        4177 non-null   object 
     19  Cabin_Num         4177 non-null   object 
     20  Cabin_Side        4177 non-null   object 
    dtypes: float64(11), object(10)
    memory usage: 701.8+ KB



```python
# Series to collate mean absolute errors for each algorithm
mae_compare = pd.Series()
mae_compare.index.name = 'Algorithm'

features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log', 'Cabin_Deck', 'Cabin_Side']


train_X = train_final.drop('Transported', axis=1)
train_y = train_final[['Transported']]

X_test = pd.get_dummies(test[features])

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

rf_model.fit(train_X, train_y)
predictions = rf_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

```

    Your submission was successfully saved!


    /Users/mikeyang/Code/Notebooks/.venv/lib/python3.14/site-packages/sklearn/base.py:1365: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


## Improvements for future

- I took the liberty to make assumptions during the data cleaning section. For the following features, I can try to predict the values based off spending of populated features.
    - VIP
    - HomePlanet
    - Destination
    - ShoppingMall
    - FoodCourt
    - RoomService
    - Spa
    - VRDeck



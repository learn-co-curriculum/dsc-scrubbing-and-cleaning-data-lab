
# Scrubbing and Cleaning Data - Lab

## Introduction

In the previous labs, you joined the data from our separate files into a single DataFrame.  In this lab, you'll scrub the data to get it ready for exploration and modeling!

## Objectives

You will be able to:

* Cast columns to the appropriate data types
* Identify and deal with null values appropriately
* Remove unnecessary columns
* Understand how to normalize data


## Getting Started

You'll find the resulting dataset from our work in the _Obtaining Data_ Lab stored within the file `walmart_data_not_cleaned.csv`.  

In the cells below:

* Import pandas and set the standard alias
* Import numpy and set the standard alias
* Import matplotlib.pyplot and set the standard alias
* Import seaborn and set the alias `sns` (this is the standard alias for seaborn)
* Use the ipython magic command to set all matplotlib visualizations to display inline in the notebook
* Load the dataset stored in the .csv file into a DataFrame using pandas
* Inspect the head of the DataFrame to ensure everything loaded correctly


```python
# Import statements go here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Now, load in the dataset and inspect the head to make sure everything loaded correctly
df = pd.read_csv("Lego_data_merged.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>ages</th>
      <th>piece_count</th>
      <th>set_name</th>
      <th>prod_desc</th>
      <th>prod_long_desc</th>
      <th>theme_name</th>
      <th>country</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>review_difficulty</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75823</td>
      <td>6-12</td>
      <td>277</td>
      <td>Bird Island Egg Heist</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Angry Birds™</td>
      <td>US</td>
      <td>$29.99</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>Average</td>
      <td>4.5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75822</td>
      <td>6-12</td>
      <td>168</td>
      <td>Piggy Plane Attack</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Angry Birds™</td>
      <td>US</td>
      <td>$19.99</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>Easy</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75821</td>
      <td>6-12</td>
      <td>74</td>
      <td>Piggy Car Escape</td>
      <td>Chase the piggy with lightning-fast Chuck and ...</td>
      <td>Pitch speedy bird Chuck against the Piggy Car....</td>
      <td>Angry Birds™</td>
      <td>US</td>
      <td>$12.99</td>
      <td>11.0</td>
      <td>4.3</td>
      <td>Easy</td>
      <td>4.3</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21030</td>
      <td>12+</td>
      <td>1032</td>
      <td>United States Capitol Building</td>
      <td>Explore the architecture of the United States ...</td>
      <td>Discover the architectural secrets of the icon...</td>
      <td>Architecture</td>
      <td>US</td>
      <td>$99.99</td>
      <td>23.0</td>
      <td>3.6</td>
      <td>Average</td>
      <td>4.6</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21035</td>
      <td>12+</td>
      <td>744</td>
      <td>Solomon R. Guggenheim Museum®</td>
      <td>Recreate the Solomon R. Guggenheim Museum® wit...</td>
      <td>Discover the architectural secrets of Frank Ll...</td>
      <td>Architecture</td>
      <td>US</td>
      <td>$79.99</td>
      <td>14.0</td>
      <td>3.2</td>
      <td>Challenging</td>
      <td>4.6</td>
      <td>4.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    10870



## Starting our Data Cleaning

To start, you'll deal with the most obvious issue: data features with the wrong data encoding.

### Checking Data Types

In the cell below, use the appropriate method to check the data type of each column. 


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10870 entries, 0 to 10869
    Data columns (total 14 columns):
    prod_id              10870 non-null int64
    ages                 10870 non-null object
    piece_count          10870 non-null int64
    set_name             10870 non-null object
    prod_desc            10512 non-null object
    prod_long_desc       10870 non-null object
    theme_name           10870 non-null object
    country              10870 non-null object
    list_price           10870 non-null object
    num_reviews          9449 non-null float64
    play_star_rating     9321 non-null float64
    review_difficulty    9104 non-null object
    star_rating          9449 non-null float64
    val_star_rating      9301 non-null float64
    dtypes: float64(4), int64(2), object(8)
    memory usage: 1.2+ MB


Now, investigate some of the unique values inside of the `list_price` column.


```python
df.list_price.unique()[:5]
```




    array(['$29.99', '$19.99', '$12.99', '$99.99', '$79.99'], dtype=object)



### Numerical Data Stored as Strings

A common issue to check for at this stage is numeric columns that have accidentally been encoded as strings. For example, you should notice that the `list_price` column above is currently formatted as a string and contains a proceeding '$'. Remove this and convert the remaining number to a `float` so that you can later model this value. After all, your primary task is to generate model to predict the price.

> Note: While the data spans a multitude of countries, assume for now that all prices have been standardized to USD.


```python
df.list_price = df.list_price.map(lambda x: float(x.replace('$', ''))) #Strip the $ sign and convert to float
#Could also potentially take advantage of str indexing but this would be less flexible and prone to potential errors /
#if the data is not consistently formatted
df.list_price.unique()[:5]
```




    array([29.99, 19.99, 12.99, 99.99, 79.99])




```python
df.info() #Note that list_price is now a float object
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10870 entries, 0 to 10869
    Data columns (total 14 columns):
    prod_id              10870 non-null int64
    ages                 10870 non-null object
    piece_count          10870 non-null int64
    set_name             10870 non-null object
    prod_desc            10512 non-null object
    prod_long_desc       10870 non-null object
    theme_name           10870 non-null object
    country              10870 non-null object
    list_price           10870 non-null float64
    num_reviews          9449 non-null float64
    play_star_rating     9321 non-null float64
    review_difficulty    9104 non-null object
    star_rating          9449 non-null float64
    val_star_rating      9301 non-null float64
    dtypes: float64(5), int64(2), object(7)
    memory usage: 1.2+ MB


### Detecting and Dealing With Null Values

Next, it's time to check for null values. How to deal with the null values will be determined by the columns containing them, and how many null values exist in each.  
 
In the cell below, get a count of how many null values exist in each column in the DataFrame. 


```python
df.isna().sum() #Could also simply recheck df.info() above which states the number of non-null entries
```




    prod_id                 0
    ages                    0
    piece_count             0
    set_name                0
    prod_desc             358
    prod_long_desc          0
    theme_name              0
    country                 0
    list_price              0
    num_reviews          1421
    play_star_rating     1549
    review_difficulty    1766
    star_rating          1421
    val_star_rating      1569
    dtype: int64



Now, get some descriptive statistics for each of the columns. You want to see where the minimum and maximum values lie.  


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>piece_count</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.087000e+04</td>
      <td>10870.000000</td>
      <td>10870.000000</td>
      <td>9449.000000</td>
      <td>9321.000000</td>
      <td>9449.000000</td>
      <td>9301.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.181634e+04</td>
      <td>503.936431</td>
      <td>67.309137</td>
      <td>17.813737</td>
      <td>4.355413</td>
      <td>4.510319</td>
      <td>4.214439</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.736390e+05</td>
      <td>831.209318</td>
      <td>94.669414</td>
      <td>38.166693</td>
      <td>0.617272</td>
      <td>0.516463</td>
      <td>0.670906</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.300000e+02</td>
      <td>1.000000</td>
      <td>2.272400</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.800000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.112300e+04</td>
      <td>97.000000</td>
      <td>21.899000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.300000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.207350e+04</td>
      <td>223.000000</td>
      <td>36.587800</td>
      <td>6.000000</td>
      <td>4.500000</td>
      <td>4.600000</td>
      <td>4.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.124800e+04</td>
      <td>556.000000</td>
      <td>73.187800</td>
      <td>14.000000</td>
      <td>4.800000</td>
      <td>5.000000</td>
      <td>4.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000431e+06</td>
      <td>7541.000000</td>
      <td>1104.870000</td>
      <td>367.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now that you have a bit more of a understanding of each of these features you can now make an informed decision about the best strategy for dealing with the various null values. 

Some common strategies for filling null values include:
* Using the mean of the feature
* Using the median of the feature
* Inserting a random value from a normal distribution with the mean and std of the feature
* Binning

Given that most of the features with null values concern user reviews of the lego set, it is reasonable to wonder whether there is strong correlation between these features in the first place. Before proceeding, take a minute to investigate this hypothesis.


```python
import itertools
```


```python
# Investigate whether multicollinearity exists between the review features 
# (num_reviews, play_star_rating, star_rating, val_star_rating)
feats = ['num_reviews', 'play_star_rating', 'star_rating', 'val_star_rating']
for combo in itertools.combinations(feats, 2):
    x = combo[0]
    y = combo[1]
    temp = df[(~df[x].isnull())&(~df[y].isnull())]
    corr = round(np.corrcoef(temp[x],temp[y])[0][1], 2)
    print("Correlation between {} and {}: {}".format(x,y, corr))
```

    Correlation between num_reviews and play_star_rating: -0.06
    Correlation between num_reviews and star_rating: 0.0
    Correlation between num_reviews and val_star_rating: 0.03
    Correlation between play_star_rating and star_rating: 0.62
    Correlation between play_star_rating and val_star_rating: 0.48
    Correlation between star_rating and val_star_rating: 0.73


 

Note that there is substantial correlation between the `play_star_rating`, `star_rating` and `val_star_rating`. While this could lead to multicollinearity in your eventual regression model, it is too early to clearly determine this at this point. Remember that multicollinearity is a relationship between 3 or more variables while correlation simply investigates the relationship between two variables.

Additionally, these relationships provide an alternative method for imputing missing values: since they appear to be correlated, you could use these features to help impute missing values in the others features. For example, if you are missing the star_rating for a particular row but have the val_star_rating for that same entry, it seems reasonable to assume that it is a good estimate for the missing star_rating value as they are highly correlated. That said, doing so does come with risks; indeed you would be further increasing the correlation between these features which could further provoke multicollinearity in the final model.

Investigate if you could use one of the other star rating features when one is missing. How many rows have one of `play_star_rating`, `star_rating` and `val_star_rating` missing, but not all three.


```python
print('Number missing all three:',
      len(df[(df.play_star_rating.isnull())
      & (df.star_rating.isnull())
      & (df.val_star_rating.isnull())])
     )
```

    Number missing all three: 1421


Well, it seems like when one is missing, the other two are also apt to be missing. While this has been a bit of an extended investigation, simply go ahead and fill the missing values with that features median.  

Fill in the missing `review_difficulty` values with 'unknown'.


```python
for col in df.columns:
    try:
        median = df[col].median()
        df[col] = df[col].fillna(value=median)
    except:
        continue
df.review_difficulty = df.review_difficulty.fillna('unknown')
df.isna().sum()
```




    prod_id                0
    ages                   0
    piece_count            0
    set_name               0
    prod_desc            358
    prod_long_desc         0
    theme_name             0
    country                0
    list_price             0
    num_reviews            0
    play_star_rating       0
    review_difficulty      0
    star_rating            0
    val_star_rating        0
    dtype: int64



## Normalizing the Data

Now, you'll need to convert all of our numeric columns to the same scale by **_normalizing_** our dataset.  Recall that you normalize a dataset by converting each numeric value to it's corresponding z-score for the column, which is obtained by subtracting the column's mean and then dividing by the column's standard deviation for every value. 


In the cell below:

* Normalize the numeric X features by subtracting the column mean and dividing by the column standard deviation. 
(Don't bother to normalize the list_price as this is the feature you will be predicting.)


```python
def norm_feat(series):
    return (series - series.mean())/series.std()
for feat in ['piece_count', 'num_reviews', 'play_star_rating', 'star_rating', 'val_star_rating']:
    df[feat] = norm_feat(df[feat])
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>piece_count</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>10870.000000</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
      <td>1.087000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.181634e+04</td>
      <td>1.115789e-16</td>
      <td>67.309137</td>
      <td>3.132256e-16</td>
      <td>3.548841e-14</td>
      <td>2.524610e-13</td>
      <td>-1.584535e-13</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.736390e+05</td>
      <td>1.000000e+00</td>
      <td>94.669414</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.300000e+02</td>
      <td>-6.050659e-01</td>
      <td>2.272400</td>
      <td>-4.264402e-01</td>
      <td>-5.883334e+00</td>
      <td>-5.641909e+00</td>
      <td>-5.193413e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.112300e+04</td>
      <td>-4.895715e-01</td>
      <td>21.899000</td>
      <td>-3.705846e-01</td>
      <td>-4.810100e-01</td>
      <td>-4.602216e-01</td>
      <td>-3.650101e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.207350e+04</td>
      <td>-3.379852e-01</td>
      <td>36.587800</td>
      <td>-2.868011e-01</td>
      <td>2.160641e-01</td>
      <td>1.615809e-01</td>
      <td>1.178302e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.124800e+04</td>
      <td>6.263593e-02</td>
      <td>73.187800</td>
      <td>-1.192341e-01</td>
      <td>5.646012e-01</td>
      <td>7.833834e-01</td>
      <td>6.006705e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000431e+06</td>
      <td>8.466055e+00</td>
      <td>1104.870000</td>
      <td>9.795146e+00</td>
      <td>1.087407e+00</td>
      <td>9.906510e-01</td>
      <td>1.244458e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['prod_id', 'ages', 'piece_count', 'set_name', 'prod_desc',
           'prod_long_desc', 'theme_name', 'country', 'list_price', 'num_reviews',
           'play_star_rating', 'review_difficulty', 'star_rating',
           'val_star_rating'],
          dtype='object')



## Saving Your Results

While you'll once again practice one-hot encoding as you would to preprocess data before fitting a model, saving such a reperesentation of the data will eat up additional disk space. After all, a categorical variable with 10 bins will be transformed to 10 seperate features when passed through `pd.get_dummies()`. As such, while the further practice is worthwhile, save your DataFrame as is for now.


```python
df.to_csv("Lego_dataset_cleaned.csv", index=False)
```

## One-Hot Encoding Categorical Columns

As a final step, you'll need to deal with the categorical columns by **_one-hot encoding_** them into binary variables via the `pd.get_dummies()` method.  

When doing this, you should also subset to appropriate features. If you were to simply pass the entire DataFrame to the `pd.get_dummies()` method as it stands now, then you would end up with unique features for every single product description! (Presumably the descriptions are unique.) As such, you should first subset to the numeric features that you will eventually use in a model along with categorical variables that are not unique.

In the cell below, subset to the appropriate predictive features and then use the [`pd.get_dummies()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) to one-hot encode the dataset.


```python
feats = ['ages', 'piece_count', 'theme_name', 'country', 'list_price', 'num_reviews',
         'play_star_rating', 'review_difficulty', 'star_rating', 'val_star_rating']
#Don't include prod_id, set_name, prod_desc, or prod_long_desc; they are too unique
df = df[feats]
df = pd.get_dummies(df)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece_count</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>play_star_rating</th>
      <th>star_rating</th>
      <th>val_star_rating</th>
      <th>ages_10+</th>
      <th>ages_10-14</th>
      <th>ages_10-16</th>
      <th>ages_10-21</th>
      <th>...</th>
      <th>country_NZ</th>
      <th>country_PL</th>
      <th>country_PT</th>
      <th>country_US</th>
      <th>review_difficulty_Average</th>
      <th>review_difficulty_Challenging</th>
      <th>review_difficulty_Easy</th>
      <th>review_difficulty_Very Challenging</th>
      <th>review_difficulty_Very Easy</th>
      <th>review_difficulty_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.273020</td>
      <td>29.99</td>
      <td>-0.398512</td>
      <td>-0.655279</td>
      <td>-0.045687</td>
      <td>-0.365010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.404154</td>
      <td>19.99</td>
      <td>-0.398512</td>
      <td>-0.655279</td>
      <td>0.990651</td>
      <td>-0.365010</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.517242</td>
      <td>12.99</td>
      <td>-0.147162</td>
      <td>-0.132473</td>
      <td>-0.460222</td>
      <td>-0.204063</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.635296</td>
      <td>99.99</td>
      <td>0.187972</td>
      <td>-1.352353</td>
      <td>0.161581</td>
      <td>0.117830</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.288812</td>
      <td>79.99</td>
      <td>-0.063378</td>
      <td>-2.049427</td>
      <td>0.161581</td>
      <td>-0.204063</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>



That's it! You've now successfully scrubbed your dataset--you're now ready for data exploration and modeling!

## Summary

In this lesson, you learned gain practice with data cleaning by:

* Casting columns to the appropriate data types
* Identifying and deal with null values appropriately
* Removing unnecessary columns
* Checking for and deal with multicollinearity
* Normalizing your data

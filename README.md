
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
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```


```python
# Now, load in the dataset and inspect the head to make sure everything loaded correctly
df = pd.read_csv('Lego_data_merged.csv')
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
df.list_price.value_counts()[:5]
```




    $24.3878    565
    $36.5878    520
    $12.1878    515
    $18.2878    304
    $42.6878    234
    Name: list_price, dtype: int64



### Numerical Data Stored as Strings

A common issue to check for at this stage is numeric columns that have accidentally been encoded as strings. For example, you should notice that the `list_price` column above is currently formatted as a string and contains a proceeding '$'. Remove this and convert the remaining number to a `float` so that you can later model this value. After all, your primary task is to generate model to predict the price.

> Note: While the data spans a multitude of countries, assume for now that all prices have been standardized to USD.


```python
#Your code here; extract the list_price as a floating number
df.list_price = df.list_price.map(lambda x : float(x[1:]))
df.list_price.unique()[:5]
```




    array([29.99, 19.99, 12.99, 99.99, 79.99])



 

 

### Detecting and Dealing With Null Values

Next, it's time to check for null values. How to deal with the null values will be determined by the columns containing them, and how many null values exist in each.  
 
In the cell below, get a count of how many null values exist in each column in the DataFrame. 


```python
df.info() #Same as checking the type above
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
      <td>9449.000000</td>
      <td>9321.000000</td>
      <td>9449.000000</td>
      <td>9301.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.181634e+04</td>
      <td>503.936431</td>
      <td>17.813737</td>
      <td>4.355413</td>
      <td>4.510319</td>
      <td>4.214439</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.736390e+05</td>
      <td>831.209318</td>
      <td>38.166693</td>
      <td>0.617272</td>
      <td>0.516463</td>
      <td>0.670906</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.300000e+02</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.800000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.112300e+04</td>
      <td>97.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.300000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.207350e+04</td>
      <td>223.000000</td>
      <td>6.000000</td>
      <td>4.500000</td>
      <td>4.600000</td>
      <td>4.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.124800e+04</td>
      <td>556.000000</td>
      <td>14.000000</td>
      <td>4.800000</td>
      <td>5.000000</td>
      <td>4.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000431e+06</td>
      <td>7541.000000</td>
      <td>367.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now that you have a bit more of a understanding of each of these features you can now make an informed decision about the best strategy for dealing with the various null values. 

* The data contained within each column are continuously-valued floats. 
* The range is quite large, with the smallest value being around 0 or even negative in some columns, and the max being greater than 100,000.
* There is extremely high variance in each, with the standard deviation being larger than the mean in all 5 columns. 


### Dealing With Null Values Through Binning

This suggests that the best bet is to bin the columns.
For now, start with with 5 bins of equal size. 

In the cell below: 

* Create a binned version of each `MarkDown` column and add them to the DataFrame.  
* When calling `pd.cut()`, pass in the appropriate column as the object to be binned, the number of bins we want, `5`, and set the `labels` parameter to `bins`, so that you have clearly labeled names for each bin. 

For more information on how to bin these columns using pd.cut, see the [pandas documentation for this method.](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html)


```python
bins = ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%']

for i in range (1, 6):
    df["binned_markdown_" + str(i)] = None
```

Great! Now, check the `.dtypes` attribute of the DataFrame to see that these new categorical columns have been created. 

They exist! However, they still contain null values.  You need to replace all null values with a string that will represent all missing values.  Use the `replace()` method or the `fillna()` method on each column and replace `NaN` with `"NaN"`. 

In the cell below, replace all missing values inside our `binned_markdown` columns with the string `"NaN"`.

**_NOTE:_** If you're unsure of how to do this, check the [pandas documentation for replace](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html).


```python
for i in range (1,6):
    None
```

Great! Now, check if those columns still contain null values. 

In the cell below, display the number of null values contained within each column of our DataFrame.

Excellent! You've now dealt with all the null values in the dataset through **_Coarse Classification_** by binning the data and treating null values as a distinct category. All that's left to do is to drop our original `MarkDown` columns from the DataFrame. 

Note that in this step, you'll also drop the `Date` column, because you are going to build a generalized model and will not be making use of any time series data. 

In the cell below:

* Create a list called `to_drop` that contains the name of every `MarkDown` column you need to drop (for a challenge, try doing this with a list comprehension!)
* Append `"Date"` to `to_drop`
* Drop these columns (in place) from the DataFrame
* Display the number of null values in each column again to confirm that these columns have been dropped, and that the DataFrame now contains no missing values



```python
to_drop = None
```

### Checking for Multicollinearity


Before you one-hot encode the categorical columns usin `pd.get_dummies()`, you'll want to quickly check the dataset for multicollinearity, since this can severly impact model stability and interpretability.  You want to make sure that the columns within the dataset are not highly correlated. 

A good way to check for multicollinearity between features is to create a correlation heatmap.

The [seaborn documentation](https://seaborn.pydata.org/examples/many_pairwise_correlations.html) provides some great code samples to help you figure out how to display a Correlation Heatmap.  

Check out this documentation, and then modify the code included below so that it displays a Correlation Heatmap for your dataset below.


```python
# Set the style of the visualization
sns.set(style="white")

# Create a covariance matrix
corr = None

# Generate a mask the size of our covariance matrix
mask = None
mask[np.triu_indices_from(mask)] = None

# Set up the matplotlib figure
f, ax = None

# Generate a custom diverging colormap
cmap = None

# Draw the heatmap with the mask and correct aspect ratio

```

Interpret the Correlation Heatmap you created above to answer the following questions:

Which columns are highly correlated with the target column our model will predict?  Are any of our predictor columns highly correlated enough that we should consider dropping them?  Explain your answer.

Write your answer below this line:
________________________________________________________________________________________________________________________________



## Normalizing the Data

Now, you'll need to convert all of our numeric columns to the same scale by **_normalizing_** our dataset.  Recall that you normalize a dataset by converting each numeric value to it's corresponding z-score for the column, which is obtained by subtracting the column's mean and then dividing by the column's standard deviation for every value. 

Since you only have 4 columns containing numeric data that needs to be normalized, you can do this by hand in the cell below. This avoids errors that stem from trying to normalize datasets that contain strings in all of our categorical columns. Plus, it's good practice to help remember how normalization works!

In the cell below:

* Normalize the following columns individually: `Size`, `Temperature`, `Fuel_Price`, `CPI`, and `Unemployment` by subtracting the column mean and dividing by the column standard deviation. 


```python
df.Size = None
df.Temperature = None
df.Fuel_Price = None
df.CPI = None
df.Unemployment = None
```

## One-Hot Encoding Categorical Columns

As a final step, you'll need to deal with the categorical columns by **_one-hot encoding_** them into binary variables via the `pd.get_dummies()` method.  

In the cell below, use the [`pd.get_dummies()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) to one-hot encode the dataset.


```python
df = None
```


```python
df.head()
```

That's it! You've now successfully scrubbed your dataset--you're now ready for data exploration and modeling!

## Summary

In this lesson, you learned gain practice with data cleaning by:

* Casting columns to the appropriate data types
* Identifying and deal with null values appropriately
* Removing unnecessary columns
* Checking for and deal with multicollinearity
* Normalizing your data

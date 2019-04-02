
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

```


```python
# Now, load in the dataset and inspect the head to make sure everything loaded correctly
df = None
```

Great! Everything looks just like we left it as.  Now, you can begin cleaning the data. 


## Dealing with Oversized Datasets

This dataset is quite large. Often, when starting out on a project, its a good idea to build the model on a subset of the data so that you're not bogged down by large runtimes. 

In the cell below, check how many rows this dataset contains.

This dataset contains `421570` rows! That's large enough that you should consider building our model on a subset of the data to increase our speed during the modeling step.  Modeling is an iterative process, and you'll likely have to fit out model multiple times as you tweak it--by subsetting the dataset, you'll protect yourself from insane runtimes every time you make a small change and need to rerun the model. Once you have a prototype built, you can always add all the extra data back in!

### Subsetting our Dataset

The typical method for subsetting our dataset is to just take a random sample of data.  This is an option for us.  However, when we inspect the columns of our dataset in a bit, we'll notice that we have 2 categorical columns with very high cardinality--`Store`, and `Dept`.  This provides us with an opportunity to reduce dimensionality while subsampling.  Instead of building a model on all the stores and departments in our dataset, we'll subset our data so that it only contains stores 1 through 10.  

In the cell below, slice our dataset so that only rows with a `Store` value between 1 and 10 (inclusive) remain. 


```python
df = None
```


```python
len(df)
```

## Starting our Data Cleaning

To start, you'll deal with the most obvious issue: data features with the wrong data encoding.

### Checking Data Types

In the cell below, use the appropriate method to check the data type of each column. 

Now, investigate the unique values inside of the `Store` and `Dept` columns.

In the cells below, use the appropriate DataFrame method to display all the unique values in the `Store` column, and in the `Dept` column.

### Categorical Data Stored as Integers

A common issue to check for at this stage is numeric columns that have accidentally been encoded as strings.  However, in this dataset, you should notice that although the `Store` and `Dept` columns are both encoded as integer values, they are categorical data representing specific stores or departments.  As such, you'll want to convert these columns to strings, so that you can then use the `pd.get_dummies()` method to create binary dummy variables. This representation, binary dummy variables, is the most appropriate encoding mechanism for categorical data when then feeding the dataset into many machine learning algorithms such as simple linear regression. If left with numeric encoding, a model would interpret Store 2 as twice Store 1 and half of Store 4.  These sorts of mathematical relationships don't make sense when integers are used to as identifiers for categories.  

In the cell below, cast the `Store` and `Dept` columns to strings. 

### Numeric Data Stored as Strings

It looks like there are two columns that are already encoded as strings (remember, pandas denotes string columns as `object`)--`Date` and `Type`.

For now, don't worry about `Date`. Quickly check out the `Type` column just to ensure that it doesn't contain numeric data.

In the cell below, get the unique values contained within the `Type` column. 

Great job--the `Type` column is clearly a categorical column. You'll first need to deal with the null values but can then use the `pd.get_dummies()` again.

Double check the column encodings again as a sanity check to make sure that everything you did above is reflected in the dataset.

### Detecting and Dealing With Null Values

Next, it's time to check for null values. How to deal with the null values will be determined by the columns containing them, and how many null values exist in each.  
 
In the cell below, get a count of how many null values exist in each column in the DataFrame. 

  

Now, get some descriptive statistics for each of the columns. You want to see where the minimum and maximum values lie.  

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



## Normalizing our Data

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

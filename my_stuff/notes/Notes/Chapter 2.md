End-to-End Machine Learning Project

Main checklist for a typical machine learning project: (more detailed than [[Studying/Self/HOMLP/Notes/Chapter 1#Typical workflow|Chapter 1: Typical workflow]])

1. Frame the problem and look at the big picture.
2. Get the data.
3. Explore the data to gain insights.
4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.
5. Explore many different models and short-list the best ones.
6. Fine-tune your models and combine them into a great solution.
7. Present your solution.
8. Launch, monitor, and maintain your system.

The example project tackled in this chapter will be to build a model of housing prices in the state of California based on California census data.
# Working with real data

Open data repositories you can use to get data:

- [Google Datasets Search](https://datasetsearch.research.google.com) 
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) 
- [OpenML.org](https://openml.org) 
- [Kaggle.com](https://kaggle.com/datasets)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data)
- [Amazon's AWS datasets](https://registry.opendata.aws)
- [U.S. Government's Open Data](https://data.gov)
- [DataPortals.org](https://dataportals.org)
- [Wikipedia's list of machine learning datasets](https://homl.info/9)


# Look at the Big Picture

Try to express the problem clearly, for example:

>Your first task is to use California census data to build a model of housing prices in the state.

Now what sort of data is that?

>This data includes metrics such as the population, median income, and median housing price for each block group in California.

Make sure everything is well defined.

>Block groups are the smallest geographical unit for which the US Census Bureau publishes ample data (a block group typically has a population of 600 to 3,000 people). I will call them “districts” for short.

Okay, now given this task and this data what's the aim?

>Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

---

Always start with a machine learning project checklist. Use the one at https://homl.info/checklist, adapt it to your needs as time goes on.


## Frame the Problem

- Ask your boss what exactly the **business objective** is.

You don't sell machine learning models, you sell solutions to problems, and a model may or may not be part of the solution.

==How does the company expect to use and benefit from this model==? Knowing the objective is important because it will determine how you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.

Your boss answers that your model’s output (a prediction of a district’s median housing price) will be essential to ==determine whether it is worth investing in a given area==. 
More specifically, your ==model’s output will be fed to another machine learning system==, along with some other signals (a piece of information fed to a ml system). 
==So it’s important to make our housing price model as accurate as we can.==

![][ml-pipeline.png]

### Pipelines

A *data pipeline* is a sequence of data processing components, these are very common in machine learning systems.
Components typically run asynchronously, each component pulls in a large amount of data, processes it and spits out the result in another data store.
Then, some time later, the next component in the pipeline pulls this data and  spits out its own output.

Each component is fairly self-contained, the interface between components is simply the data store. (very modular and robust architecture)

If a component breaks down, the downstream components can often continue to run normally (at least for a while) by just using the last output from the broken component. (however broken components can go unnoticed for some time)

---

- Next ask what the **current solution** looks like (if any).

The current situation will give you a ==reference for performance== as well as ==insights on how to solve the problem==.

Your boss answers that the district housing prices are currently ==estimated manually by experts==: a ==team gathers up-to-date information about a district==, and ==when they cannot get the median housing price, they estimate it using complex rules==.

This is ==costly and time-consuming==, and ==their estimates are not great==; in cases where they manage to find out the actual median housing price, they often realize that their ==estimates were off by more than 30%==.

This is why the company thinks that it would be useful to train a model to predict a district’s median housing price, given other data about that district. The census data looks like a great dataset to exploit for this purpose, since it includes the median housing prices of thousands of districts, as well as other data.

---

Determine what kind of task this is (classification, regression...), the type of training supervision, how it should get its data?

The model you'd pick for this would probably be *supervised* (labeled examples). 
It's a typical *regression* task (predicting a value). 

More specifically, this is a *multiple regression* problem, since the system will use multiple features to make a prediction (the district’s population, the median income, etc.). 

It is also a *univariate regression* problem, since we are only trying to predict a single value for each district.

There is no continuous flow of data coming into the system, and the data is small enough to fit in memory, you can use *batch learning*.

## Select a performance measure

A typical performance measure for regression problems is the *root mean squared error* (*RMSE*).
It gives an idea of how much error the system typically makes in its predictions, with a higher weight given to large errors.

$$

RMSE(\mathbf{X}, y, h) = \sqrt{ \frac{1}{m} \sum_{i=1}^m ( h( \mathbf{X}^{(i)} ) - y^{(i)} )^2 }

$$

See [[Notations|Notations]] for details on this equation and common notation for ml.

Basically computes the average difference between the values the system predicted and the real life stuff, squares it then takes the root so that negative and positive differences don't cancel each other out, both are mistakes, both should count.
Somewhat similar to how the standard deviation is computed.

---

Although the RMSE is generally the preferred performance measure for regression tasks, in some contexts you may prefer to use another function, especially ==when there are many outliers in the data, as the RMSE is quite sensitive to them==.

In that case you may consider the *mean absolute error* (MAE, also called the *average absolute error*):

$$

MAE(\mathbf{X}, y, h) =  \frac{1}{m} \sum_{i=1}^m \lvert h( \mathbf{X}^{(i)} ) - y^{(i)} \rvert
$$

Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values. 

Various distance measures, or norms, are possible:

-   Computing the root of a sum of squares (RMSE) corresponds to the *Euclidean norm*: this is the notion of distance we are all familiar with. It is also called the $\ell_2$ norm, denoted $\|\cdot\|_2$ (or just $\|\cdot\|$).
  
- Computing the sum of absolutes (MAE) corresponds to the $\ell_1$ norm, denoted $\|\cdot\|_1$. This is  sometimes called the *Manhattan norm* because it measures the distance between two points in a city if you can only travel along orthogonal city blocks.

-   More generally, the $\ell_k$ norm of a vector $v$ containing $n$ elements is defined as
    $\|v\|_k = (|v_1|^k + |v_2|^k + ... + |v_n|^k)^{1/k}$. $\ell_0$ gives the number of nonzero elements in the vector, and $\ell_\infty$ gives the maximum absolute value in the vector.


The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

## Check the Assumptions

List and verify the assumptions that have been made so far (by you or others) to catch serious issues early on.

For example, the district prices that your system outputs are going to be fed to a downstream machine learning system, and you **assume** that these **prices are going to be used as such**.

**However**, what if the downstream system converts the prices into categories (e.g., "cheap", "medium", "expensive"), and then uses these categories instead of the prices themselves.

This would mean that getting the prices perfectly right is not important, further more this would frame the problem as a **classification** task instead of regression as you had previously assumed.

You don’t want to find this out after working on a regression system for months.

# Get the Data

## Download the data

Rather than manually downloading and decompressing the data, you should write a function that does it for you, so that you can write a script to use the function to fetch the latest data regularly.

Automating the process of fetching the data is also useful if you need to install the dataset on multiple machines.

---

The function below first looks for the *datasets/housing.tgz* file [1], if it does not find it, it creates the *datasets* folder [2], then it downloads the tarball from the *ageron/data* GitHub repository to the *datasets* folder [3] and extracts its contents into that directory [4].
The contents should be *datasets/housing/housing.csv*, the function loads this CSV file into a pandas DataFrame object and returns it [5].

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
	# 1
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
	    # 2
        Path("datasets").mkdir(parents=True, exist_ok=True)
        # 3
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        # 4
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    # 5
    return pd.read_csv(Path("datasets/housing/housing.csv"))
    
housing_full = load_housing_data()

```

>If you are using Python 3.12 or 3.13, you should add filter='data' to the extractall() method’s arguments: this limits what the extraction algorithm can do and improves security (see the documentation for more details).


## Take a Quick Look at the Data Structure

### Examine the data

`df` just stands for the DataFrame's name, in the previous example that would be `housing_full`

Use:

```python 
df.head()
``` 

![][housing_full.head.png]

>[!Info]-
>You'll notice that the data concerns each district, **NOT** houses individually.
>You get the total number of rooms/bedrooms/people across all households of the district, as well as the number of households in the district.
>You may want to, and you shall later, compute the average number of rooms/bedrooms/people per household in each district based on the data available to you.

to take a look at the top five rows of the DataFrame. For instance here each row represents one district, there are 10 attributes:
`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, and `ocean_proximity`

Use:

```python
df.info()
```

to get a description of the data, including the number of rows, the number of columns, each attribute's type, the number of non-null values etc.

```python 
>>> housing_full.info()
# Output on the CLI, not code:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 longitude 20640 non-null float64
1 latitude 20640 non-null float64
2 housing_median_age 20640 non-null float64
3 total_rooms 20640 non-null float64
4 total_bedrooms 20433 non-null float64
5 population 20640 non-null float64
6 households 20640 non-null float64
7 median_income 20640 non-null float64
8 median_house_value 20640 non-null float64
9 ocean_proximity 20640 non-null object
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
```

Notice that there are 20640 entries in total, but that for `total_bedrooms` there are only 20433 non-null entries, you will need to take care of the 207 districts that are missing this feature.

`ocean_proximity` is the only non-numerical data type, it is a Python object, but since you loaded the data from a CSV you know that it must be a text attribute.
By looking at the first few rows you'll notice that the values in that columns are repetitive, which means that it is probably a categorical attribute.
You can find out what categories exist and how many districts belong to each category by using the method `df["Column Name"].value_counts()` :

``` python
>>> housing_full["ocean_proximity"].value_counts()
ocean_proximity
<1H OCEAN 9136
INLAND 6551
NEAR OCEAN 2658
NEAR BAY 2290
ISLAND 5
Name: count, dtype: int64
```

The method:

```python
df.describe()
```

shows a summary of the numerical attributes, with the `count` (non-null values), `mean`, `std`, `min`, `25%`, `50%`, `75%` (percentiles, value for which `x%` of the observations are below) and the `max` for each and every numerical column.

### Plot the data

Another quick way to get a feel of the type of data you are dealing with is to plot a
histogram for each numerical attribute.
You can either plot one attribute at a time, or use the `df.hist()` method for the whole dataset (will only plot numerical values).

```python
import matplotlib.pyplot as plt

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12, 8))

plt.show()
```

Just load your data onto a DataFrame, call `hist` then `plt.show()` and you already have an idea about what the data looks like.

The number of value ranges can be adjusted using the bins argument, the larger it is the more granular the histogram, but values tend to  sort of 'cluster', they get pushed back by very large outliers as the histogram tries to represent them by changing the scale of the chart.

Also with a smaller number of bins it's easier to see when data has been 'clamped' to be between a range, read below.

---

![][histograms.png]

Looking at the histograms you notice a few things:

- The median income is not expressed in dollars, by asking the team that collected the data you may learn that it has been scaled and capped at 15.0001 for higher median incomes, and 0.49999 for lower median incomes (clamped).
  The numbers represent roughly tens of thousands of dollars (with a range of around $5000 to $150,000).
  Working with preprocessed attributes is common in machine learning, and it is not necessarily a problem, but you should try to understand how the data was computed.
- The housing median age and the housing median were also capped.
  This means there's no real data for houses that are more expensive than $500,000, your machine learning algorithm may learn that prices never go beyond that limit.
  You need to check with your client team (the team that will use your system’s output) to see if this is a problem or not. If they tell you that they need precise predictions even beyond $500,000, then you have two options:
	- Collect proper labels for the districts whose labels were capped. (basically getting the actual data, but from where?)
	- Remove those districts from the training set (and also from the test set, since your system should not be evaluated poorly if it predicts values beyond $500,000)
- These attributes have very different scales, this'll be explained more later.
- Many histograms are skewed *right*, this may make it a bit arder for some machine learning algorithms to detect patterns. You'll need to transform these attributes to have more symmetrical/bell-shaped distributions.


You should now have a better understanding of the kind of data you’re dealing with.
## Create a Test Set

**Before** you look at the data any further you need to create a test set, put it aside and never look at it.
Your very own human brain is amazing at pattern detection, which means it is prone to overfitting: if you look at the test set you may stumble upon some seemingly interesting pattern in the test data that leads you to select a particular kind of machine learning model.
When you estimate the generalization error using the test set your estimate will be too optimistic, your model will perform well on the test set, but not irl.
This is called *data snooping* bias.

---

### Shuffle the data

To create a test set just shuffle data, pick 20% (less if the dataset is large) of it and set it aside.

```python
import numpy as np

# 1
def shuffle_and_split_data(data, test_ratio, rng):

    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
```

Here's an example of how that function works:

```
data = [10, 2, 51, 9, 65] // test_ratio = 0.2

shuffled indices = [1, 0, 3, 2, 4]
test_indices = [1]
train_indices = [0, 3, 2, 4]

data.iloc[train_indices] = [10, 9, 51, 65]
data.iloc[test_indices] = [2]

```

You can call it like this:

```python 
>>> rng = np.random.default_rng() # default random number generator
>>> train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
>>> len(train_set)
16512
>>> len(test_set)
4128
```

### Make sure no training data **ever** ends up in the test set

Every time the program runs it'll generate a different test set, over time you (or your machine learning algorithm) will get to see the whole data set.
To solve that you can either save the test set on the first run and load it each subsequent time, or set a seed for the random number generator `np.random.default_rng(seed=67)` to ensure it always generates the same sequence of random numbers every time you run the program.


An alternative method is to compute a hash of each instance's identifier, and include that instance in the test set if the has value is less than or equal to 20% of the maximum has value.
This solution is more robust and stable when fetching an updated dataset, but does require that each instance has a unique and immutable identifier. 
What we mean by robust is that we ensure that we **never ever use any piece of data that was used in the training set in the test set**, that would ruin the entire thing.
==This ensures that the new test set will contain 20% of the new instances, but it will not contain any
instance that was previously in the training set (because ID are immutable).==
(You could use the id to construct the test set in some other way, you don't have to use hashing, **what's more robust is using IDs to determine what goes into the test set**)

Here is a possible implementation:

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
	return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
	return data.loc[~in_test_set], data.loc[in_test_set
```

The housing dataset you're using does not have an identifier column, but you can use the **row index as an id**, you still need to make sure that any new data gets appended at the end of the dataset, and that no row ever gets deleted. If that's not possible try to use **the most stable features** to build a unique identifier (latitude and longitude for instance).

Using the index as an ID:

```python
housing_with_id = housing_full.reset_index() # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
```

Using latitude and longitude as an ID

```python
housing_with_id["id"] = (housing_full["longitude"]*1000 + housing_full["latitude"])
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
```

---

Scikit-Learn provides a few functions to split datasets into multiple subsets. 
The simplest function is `train_test_split()`, which does pretty much the same thing as the `shuffle_and_split_data()` function with a few additional features.

The `random_state` parameter is essentially just a seed.
You can pass **multiple datasets** with the an **identical number of rows** to the function and it will split them on the same indices, this is useful for example if you have a separate DataFrame for labels"

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)
```

---

### Ensure the data is representative

#### Stratification

Purely random sampling methods are fine if the dataset is large enough (especially in regards to the number of attributes), but if it is not you run the risk of introducing a significant *sampling bias*.

For example if you're sampling a 1000 random people, you need to try to ensure that the Male/Female split is as close as possible to the actual percentages, there's a very high chance that you could be off by a percent or two if sampling purely randomly, that is significant. 
If you use purely random sampling, there would be over $10\%$ change of sampling a skewed test set with less than $49\%$ or more than $54\%$ female participants (using the binomial distribution). 

>[!INFO]-
> To compute that error probability you can use the binomial distribution:
> ```python
> from scipy.stats import binom
> 
> sample_size = 1000
> ratio_female = 0.516
> proba_too_small = binom(sample_size, ratio_female).cdf(490-1)
> proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)
> print(proba_too_small + proba_too_large)
> ```



Such errors only average out with a huge number of samples.

If $48.4\%$  of the population are male and $51.6\%$ are female, you need to purposefully poll $484$ males and $516$ females. 

==This is called *stratified sampling*, dividing the population into homogenous subgroups called *strata* and sampling the right number of instances from each stratum to guarantee that the test set is representative of the overall population.==

---

In our example, if you learn that the median income is a very important attribute to predict median housing prices, you'll want to ensure that the test set is representative of the various categories of incomes in the whole dataset.

Since the median income is a continuous numerical attribute, you first need to create an `income_category` attribute
By looking at the histogram you can tell that most median income values are clustered around $1.5$ to $6.0$ (i.e., $\$15,000$ to $\$60,000$), but some median incomes go far beyond $6$.

It is important to have a **sufficient number of instances** in your dataset **for each stratum**, or else the estimate of a stratum's importance may be biased.
==This means that you should not have too many strata, and each stratum should be large enough.==


The following code uses the `pd.cut()` function to create an income category attribute with five categories, labeled from $1$ to $5$.

Category ranges from $0$ to $1.5$ (i.e., less than $\$15,000$):

```python
housing_full["income_cat"] = pd.cut(housing_full["median_income"], 
									bins = [0., 1.5, 3., 4.5, 6., np.inf],
									labels=[1, 2, 3, 4, 5])
```

You can represent the categories to visualize them:

```python
cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
```

![][income_categories.png]


Now you are ready to do stratified sampling based on the income category.
Scikit-Lean provides a number of splitter classes in the `sklearn.model_selection` package that implement various strategies to split your dataset into a training set and a test set.

#### Ways to stratify (code)

Each type of splitter has a `split()` method that returns an **iterator** over different training/test splits of the same data.

==The `split()` methods yields the training and testing *indices* **NOT** the data itself.==

Having multiple splits can be useful if you want to better estimate the performance of your model, this is related to cross validation.

The following code generates 10 different stratified splits of the same dataset.

`StratifiedShuffleSplit` creates a splitter object with attributes regarding the number of splits to make, the test size and a seed[1].
You can use it as a sort of iterator when you apply the splitter. on a dataset with a column to stratify on [2].
For each iteration you get two sets of indices (I just mean set colloquially, these are DataFrames not python sets), split the data according to these indices [3] and store each training set/test set pair in an array to use them later [4].

```python
from sklearn.model_selection import StratifiedShuffleSplit

# 1
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

# 2
for train_index, test_index in splitter.split(housing_full, housing_full["income_cat"]):
	# 3
	strat_train_set_n = housing_full.iloc[train_index]
	strat_test_set_n = housing_full.iloc[test_index]
	# 4
	strat_splits.append([strat_train_set_n, strat_test_set_n])

```

Then you can use the first split:

```python
start_train_set, start_test_set = strat_splits[0]
```

==Since stratified sampling is fairly common, there’s a shorter way to get a single split using the`train_test_split()` function with the stratify argument:==

```python
strat_train_set, strat_test_set = train_test_split( 
	housing_full, 
	test_size=0.2, 
	stratify=housing_full["income_cat"], 
	random_state=42
)
```

And then to check the proportion of each stratum:

```python
>>> strat_test_set["income_cat"].value_counts() / len(strat_test_set)
income_cat
3 0.350533
2 0.318798
4 0.176357
5 0.114341
1 0.039971
Name: count, dtype: float64
```

With similar code you can measure the income category proportions in the full dataset and compare them to the test set generated with stratified sampling, as well as a dataset generated with  purely random sampling:

>[!info]- Code for computing the proportions
>```python
>def incom_cate_proportions(data):
>	return data["income_cate].value_counts() / len(strat_test_set)
>
># Purely random sampling
>train_set, test_set = train_test_split(
>	housing_full, 
>	test_size, 
>	random_state=42)
>
># Stratified sampling
>strat_train_set, strat_test_set = train_test_split( 
>	housing_full, 
>	test_size=0.2, 
>	stratify=housing_full["income_cat"], 
>	random_state=42) 
>
>compare_props = pd.DataFrame({"
>	Overall %": income_cat_proportions(housing_full),
>	"Stratified %": income_cat_proportions(strat_test_set),
>	"Random %": income_cat_proportions(test_set),
>	}).sort_index()
>
>compare_props.index.name = "Income Category"
>
>compare_props["Strat. Error %"] = (
>	compare_props["Stratified %"] / compare_props["Overall %"] - 1)
>
>compare_props["Rand. Error %"] = (
>	compare_props["Random %"] / compare_props["Overall %"] - 1)
>
>(compare_props * 100).round(2)
>```

![][sampling_bias_comparison.png]
                                  

You won’t use the `income_cat` column again, so you might as well drop it, reverting the data back to its original state:

```python
for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis=1, inplace=True)
```

==You can and should define custom columns to use when processing data and then remove them when you're done.==


# Explore and Visualize the Data to Gain Insights

Looking at a couple of histograms is no real visualization, you need to go into more depth.

First make sure you've put the test set aside and are only exploring the training set.

If the training set is very large, you may want to sample an exploration set, to make manipulations easy and fast during the exploration phase (I reckon you'd also need to stratify that).

Make a copy of the original training set so you can revert to it afterwards:

```python
housing = strat_train_set.copy()
```

## Visualizing Geographical Data

Because the dataset includes geographical information (**latitude and longitude**), it is a good idea to create a **scatterplot** of all the districts to visualize the data:

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.show()
```

![][geographical_scatterplot.png]

The data is crowded and the points are stacked on top of each other, ==to better visualize high density you can set the alpha option to $0.2$==:

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()
```

![][geographical_scatterplot_alpha.png]

You can represent more info on the same plot, the radius of each circle can represent the district's population (option `s`), the color can represent the price (option `c`), you can use a predefined color map called `jet` where low values are in blue and high values are in red.

```python
housing.plot(
	kind="scatter", 
	x="longitude", 
	y="latitude", 
	grid=True,
	s = housing["population"] / 100, label="population",
	c="median_house_value", cmap="jet", colorbar=True,
	legend=True, 
	sharex=False, 
	figsize=(10, 7)
)
plt.show()
```

![][geographical_scatterplot_alpha_color_sizes.png]

> [!note]- Pretty data
> You can make the plot prettier still by changing the labels of the axes and the color range, and also by overlaying an image of California to the background.
> You'll need to manually make sure that the latitudes and longitudes match when selecting an image.
>
> ```python
> # Download the California image
> filename = "california.png"
> filepath = Path(f"my_{filename}")
> if not filepath.is_file():
>     homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
>     url = homlp_root + "images/end_to_end_project/" + filename
>     print("Downloading", filename)
>     urllib.request.urlretrieve(url, filepath)
>  
> housing_renamed = housing.rename(columns={
>     "latitude": "Latitude", 
>     "longitude": "Longitude",
>     "population": "Population",
>     "median_house_value": "Median house value (ᴜsᴅ)"})
>     
> housing_renamed.plot(
>     kind="scatter", 
>     x="Longitude", 
>     y="Latitude",
>     s=housing_renamed["Population"] / 100, label="Population",
>     c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
>     legend=True, 
>     sharex=False, 
>     figsize=(10, 7))
>
> california_img = plt.imread(filepath)
> axis = -124.55, -113.95, 32.45, 42.05
> plt.axis(axis)
> plt.imshow(california_img, extent=axis)
>  
> plt.show()
> ```
>
> ![[california.png]]


## Look for Correlations

### Pearson's r

Since the dataset is not too large, you can easily compute the *standard correlation coefficient* (also called *Pearson’s r*) between every pair of numerical attributes using the `corr()` method:

```python
corr_matrix = housing.corr(numeric_only=True)
```

>[!info]- Pearson correlation coefficient
>From Wikipedia:
>
>In [statistics](https://en.wikipedia.org/wiki/Statistics "Statistics"), the **Pearson correlation coefficient** (**PCC**) is a [correlation coefficient](https://en.wikipedia.org/wiki/Correlation_coefficient "Correlation coefficient") that measures [linear](https://en.wikipedia.org/wiki/Linear "Linear") correlation between two sets of data. It is the ratio between the [covariance](https://en.wikipedia.org/wiki/Covariance "Covariance") of two variables and the product of their [standard deviations](https://en.wikipedia.org/wiki/Standard_deviation "Standard deviation"); thus, it is essentially a normalized measurement of the covariance, such that the result always has a value between −1 and 1.
>
>A key difference is that unlike covariance, this correlation coefficient does not have [units](https://en.wikipedia.org/wiki/Unit_of_measurement "Unit of measurement"), allowing comparison of the strength of the joint association between different pairs of random variables that do not necessarily have the same units.
>
>As with covariance itself, the measure can only reflect a linear [correlation](https://en.wikipedia.org/wiki/Correlation "Correlation") of variables, and ignores many other types of relationships or correlations. 
>As a simple example, one would expect the age and height of a sample of children from a school to have a Pearson correlation coefficient significantly greater than 0, but less than 1 (as 1 would represent an unrealistically perfect correlation).

Now you can look at how much each attribute correlates with the median house value:

```python
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value 1.000000
median_income 0.688380
total_rooms 0.137455
housing_median_age 0.102175
households 0.071426
total_bedrooms 0.054635
population -0.020153
longitude -0.050859
latitude -0.139584
Name: median_house_value, dtype: float64
```

The correlation coefficient ranges from -1 to 1.
When it is close to 1 it means that there is a strong positive correlation. (as x increases y increases as well)
When it is close to -1 it means that there is a strong negative correlation. (as x increases y decreases, and vice versa)
When it is close to 0 there is no **linear** correlation.

==There could be a polynomial/exponential/logarithmic/whatever else relation (as x approaches c -> y goes up or something) between them though, this is a limitation of this coefficient.==

Here are examples of relations where the correlation coefficient is equal to 0, despite the fact that their axes are clearly not independent. (third row)

Also it gives no information as to the slope of the relationship, only whether they both evolve in tandem or not. (second row)

![][correlation_coefficient.png]

---

### Plotting the attributes against each other
Another way to check for correlation between attributes is to use the Pandas `scatter_matrix()` function, which plots every numerical attribute against every other numerical attribute. 

Since there are now 9 numerical attributes, you would get 92 = 81 plots, which would not fit on a page, so you decide to ==focus on a few promising attributes that seem most correlated with the median housing value==.

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

plt.show()
```

![][matrix_scatter_plots.png]

The main diagonal displays a histogram of each attribute, which is more useful than plotting the variable against itself and getting a bunch of straight lines.

The most promising attribute to predict the median house value seems to be the median income, you may plot it separately to zoom in:

```python
housing.plot(
	kind="scatter",
	x="median_income",
	y="median_house_value",
	alpha=0.1,
	grid=True
)
plt.show()
```

![][income_house_value.png]

This plot reveals a few things. 
First, the correlation is indeed quite strong; you can clearly see the upward trend, although the data is noisy.
Second, the price cap you noticed earlier is clearly visible as a horizontal line at $500,000. 

==But the plot also reveals other less obvious straight lines: a horizontal line around $450,000, another around $350,000, perhaps one around $280,000, and a few more below that. ==

You may want to try removing the corresponding districts to prevent your algorithms from learning to reproduce these data quirks.

## Experiment with Attribute Combinations
In the previous sections you identified data quirks that you may want to clean up before using the data, and you found interesting correlations between attributes, in particular with the target attribute.

You also noticed that some **attributes** have a **skewed-right distribution**, so you may want to **transform them**. (e.g., by computing their logarithm or square root)

One last thing you may want to do before preparing the data for machine learning
algorithms is to try out various attribute combinations.

> [!Data]-
> 
> As mentioned previously, the columns `total_rooms`, `total_bedrooms` and `population` concern each district as a whole. 
> You'll need to compute the average numbers per household per district, or elsewise any interesting bit of data, like the ratio of bedrooms to rooms (also gives an idea about the number of bathrooms I think).

For example the total rooms in a district is not very useful if you don't know how many households there are. What you really want is the number of rooms per household.
The total number of bedrooms is also not very useful, you probably want to compare it to the total number of rooms. (*you* need to do this job, compute these values beforehand and add them to the data, don't just feed the ML raw data and expect it to figure this out)

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
```

Look at the correlation matrix again:

```python
>>> corr_matrix = housing.corr(numeric_only=True)
>>> corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value 1.000000
median_income 0.688380
rooms_per_house 0.143663 # Computed
total_rooms 0.137455 # Raw
housing_median_age 0.102175 
households 0.071426
total_bedrooms 0.054635 # Raw
population -0.020153 # Raw
people_per_house -0.038224 # Computed
longitude -0.050859
latitude -0.139584
bedrooms_ratio -0.256397 # Computed
Name: median_house_value, dtype: float64
```

The new `bedrooms_ratio` is much more correlated with the median house value than the total number of rooms or bedrooms.
It's a strong *negative* correlation, so houses with a *lower* bedroom to room ratio tend to be more expensive.


>[!Aparté]+
>
>Just throwing spaghetti at the wall here, but since attributes are useful if they tell use *something* about the label, whether the correlation be negative or positive, then we could maybe compute the average value of *r* for the raw attributes and the computed ones, and then compare them to see what's best:
>
> $\langle Raw \rangle = \frac{0.137455+0.054635+|-0.020153|}{3} = 0.070747\overline{6}$
>
> $\langle Computed \rangle = \frac{0.143663+|-0.038224|+|-0.256397|}{3} = 0.146094\overline{6}$
> 
> So if we just use the computed values that's a pretty stark improvement, but then again I think you can mix and match attributes and keep only the best, whether raw or computed, idk/

>[!warning]+
>When creating new combined features, make sure they are not too linearly correlated with existing features: collinearity can cause issues with some models, such as linear regression. In particular, avoid simple weighted sums of existing features.


# Prepare the Data for Machine Learning Algorithms

Write functions to do this, so you can easily reproduce these transformations on any dataset, gradually build a library of transformation functions, use them in the live system to transform any new data, and easily try out different combinations of transformations to see which is best.

First revert to a clean training set by copying `strat_train_set` once again (getting rid of the extra columns we made and any other transformations we did)

You should also ==separate the predictors and the labels==, since you don’t necessarily want to apply the same transformations to the predictors and the target values:

```python
# drop() returns a copy of the data and does not affect strat_train_set
# Here we're just putting everything but the column of the labels (i.e., all the fatures) into the variable housing
housing = strat_train_set.drop("median_house_value", axis=1)

# Now we're storing the column of the labels into another variable
housing_labels = strat_train_set["median_house_value"].copy()
```

Notice how we always keep `strat_train_set` unchanged, always keep a copy of the original raw data.

## Clean the data

### Missing features

You need to take care of missing features, for example the `total_bedrooms` attributes has a number of missing values, you have three options to fix this:

1. Get rid of the corresponding entries
2. Get rid of the whole attribute
3. Set the missing values to some value (zero, the mean, the median etc.). 
   This is called *imputation*.

You can use the Pandas DataFrame's methods `dropna()`, `drop()` and `fillna()` to do this:
1. 
	- `subset`: specifies which columns to consider for NA checks
	- `inplace` specifies whether to modify the original DataFrame in-place or to return a new one.  
2. 
	- `axis`: {0 or ‘index’, 1 or ‘columns’}, default 0 Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
3. 
	- `fillna()` modifies the DataFrame in place

```python
# 1
housing.dropna(subset=["total_bedrooms"], inplace=True)
# 2
housing.drop("total_bedrooms", axis=1, inplace=True) 
# 3
median = housing["total_bedrooms"].median()
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
```


> [!Aparté] 
> Remember that at this point in time we're only looking at the training set, the test set has already been put aside, it may contain NA values. 
> I think you're only supposed to impute values based on the training set, so even in the test set the mean/median/whatever used to fill NA values will have come from training set data, not the test set.

In the example we'll go with option 3 since it is the **least destructive**.
Instead of the preceding code you will use a Scikit-Learn class: `SimpleImputer`.

#### Imputers

==The benefit is that it will store the median value of each feature==, this will make it possible to impute missing values not only for the training set, but also on the validation set, the test set and any new data fed to the model.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

>[! Aparté]+
>
>Using this object instead of the pandas methods or even some custom script to apply them is better.
>It saves time, it's consistent, you can apply the same transformation multiple times, you can integrate it into workflows and pipelines with other scikit tools.
>But mostly you don't have to reinvent the wheel, all your spaghetti code will just converge to a shittier version of what scikit already does, so just use that.
>
>Also you **DO NOT** actually wanna change the median as new data arrives, your model was trained on the initial median value, changing it ruins everything. 
>Instead of scribbling it on a piece of paper, make the damn scikit object, and export it or save it somewhere or something.
>It doesn't matter if it changes, and in all cases for a large N and/or a standard distribution it wouldn't change all that much.
>
>This imputer computes medians once during its 'training' phase (`fit`), and then the values are stored in `statistics_`, you can use them to fill NA values in any of the columns of the data you fed it. As long as you don't retrain it it'll always remember those initial median values and always use them when told to impute.


The median can only be computed on numerical attributes, create a copy of the data containing only these (here this will just exclude `ocean_proximity`):

```python
housing_num = housing.select_dtypes(include=[np.number])
```

And then fit the imputer instance to the training data using the `fit()` method:

```python
imputer.fit(housing_num)
```

The imputer will compute the median of each attribute and store the result in its `statistics_` (note the underscore) instance variable.

==Only the `total_bedrooms` has missing values now, but you can't be sure that there won't be missing values in other numerical columns in any new data==, so you should apply the imputer to all the numerical attributes:

```python
>>> imputer.statistics_
array([-118.51 , 34.26 , 29. , 2125. , 434. , 1167. , 408. , 3.5385])
# Compare it to the medians computed using a pandas method
>>> housing_num.median().values
array([-118.51 , 34.26 , 29. , 2125. , 434. , 1167. , 408. , 3.5385])
```

Now you can use this trained imputer to transform the training set by replacing the missing values with the learned medians:

```python
X = imputer.transform(housing_num)
```

Other applicable strategies (arguments for `strategy`) are:
- `(strategy="mean")`
- `(strategy="most_frequent")`: Supports non-numerical data
- `(strategy="constant", fill_value=…)`: Supports non-numerical data



> [!Other imputers]
> 
> There are also more powerful imputers available in the sklearn.impute package (both for numerical features only):
> 
> - **KNNImputer** replaces each missing value with the mean of the k-nearest neighbors’ values for that feature. The distance is based on all the available features.
>
> - **IterativeImputer** trains a regression model per feature to predict the missing values based on all the other available features. It then trains the model again on the updated data, and repeats the process several times, improving the models and the replacement values at each iteration.

#### Scikit learn design - **Read this section**

Scikit-Learn’s API is remarkably well designed. These are the [main design principles](https://homl.info/11):

> [!Scikit-Learn Design]+
> All objects within scikit-learn share a uniform common basic API consisting of 
> three complementary interfaces: an estimator interface for building and fitting
> models, a predictor interface for making predictions and a transformer interface
> for converting data.

- **Consistency**: All objects share a ==consistent and simple interface== composed of a ==limited set of methods==:
  
	- *Estimators*: Any ==object that can estimate some parameters based on a dataset== is called an *estimator* (e.g., a `SimpleImputer` is an estimator).
	  The estimation itself is performed by the `fit()` method, it takes a dataset as a parameter, or two for supervised learning algorithms (the second one containing the labels).
	  Any other parameter needed to guide the estimation process, such as the imputer's strategy, is considered a *hyperparameter* and must be set as an instance variable, usually via the constructor parameter.
	  
	- *Transformers*: Some estimators can also transform a dataset, these are called <abbr title="Robots in Disguise">*transformers*</abbr>.
	  Simply call the `transform()` method with the dataset to transform as a parameter, it'll **return** the transformed dataset (doesn't modify it inplace).
	  This transformation generally relies on the learned parameters.
	  All transformers have a convenience method called `fit_transform()` which is sometimes more optimized than manually calling `fit()` and then `transform()`.
	  
	- *Predictors*: Some estimators, given a dataset, are capable of making predictions, they are called *predictors*. 
	  For example the `LinearRegression` model used in the code in Chapter 1 was a predictor, given a country's GDP it predicted life satisfaction.
	  Use the method `predict()` which takes a dataset of new instances and returns a dataset of corresponding predictions.
	  Predictors also have a `score()` method that measures the quality of the predictions given a test set (and the corresponding labels, in the case of supervised learning algorithms).
  
- **Inspection**: All the estimator’s ==hyperparameters are accessible directly via public instance variables== (e.g., `imputer.strategy`), and all the estimator’s ==learned parameters are accessible via public instance variables with an underscore suffix== (e.g., `imputer.statistics_`).
  
  
- **Non-proliferation of classes**: ==Learning algorithms are the only objects to be represented by custom classes==. ==Datasets are== represented as ==NumPy arrays or SciPy sparse matrices==, ==hyperparameters are== represented as ==strings or numbers== whenever possible.
  The only custom scikit classes you'll ever deal with are the learning algorithms.
  
- **Composition**: Existing building blocks are reused as much as possible, since many machine learning tasks are expressible as sequences/combinations of transformations to data.
   For instance you can create a `Pipeline` estimator from an arbitrary sequence of transformers followed by a final estimator.
   
- **Sensible defaults**: Scikit-Learn provides reasonable default values for most parameters, making it easy to quickly create a baseline working system.


 

















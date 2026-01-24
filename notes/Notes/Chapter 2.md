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
- [Amazon&#39;s AWS datasets](https://registry.opendata.aws)
- [U.S. Government&#39;s Open Data](https://data.gov)
- [DataPortals.org](https://dataportals.org)
- [Wikipedia&#39;s list of machine learning datasets](https://homl.info/9)

# Look at the Big Picture

Try to express the problem clearly, for example:

> Your first task is to use California census data to build a model of housing prices in the state.

Now what sort of data is that?

> This data includes metrics such as the population, median income, and median housing price for each block group in California.

Make sure everything is well defined.

> Block groups are the smallest geographical unit for which the US Census Bureau publishes ample data (a block group typically has a population of 600 to 3,000 people). I will call them “districts” for short.

Okay, now given this task and this data what's the aim?

> Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

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

- Computing the root of a sum of squares (RMSE) corresponds to the *Euclidean norm*: this is the notion of distance we are all familiar with. It is also called the $\ell_2$ norm, denoted $\|\cdot\|_2$ (or just $\|\cdot\|$).
- Computing the sum of absolutes (MAE) corresponds to the $\ell_1$ norm, denoted $\|\cdot\|_1$. This is  sometimes called the *Manhattan norm* because it measures the distance between two points in a city if you can only travel along orthogonal city blocks.
- More generally, the $\ell_k$ norm of a vector $v$ containing $n$ elements is defined as
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

> If you are using Python 3.12 or 3.13, you should add filter='data' to the extractall() method’s arguments: this limits what the extraction algorithm can do and improves security (see the documentation for more details).

## Take a Quick Look at the Data Structure

### Examine the data

`df` just stands for the DataFrame's name, in the previous example that would be `housing_full`

Use:

```python
df.head()
```

![][housing_full.head.png]

> [!Info]-
> You'll notice that the data concerns each district, **NOT** houses individually.
> You get the total number of rooms/bedrooms/people across all households of the district, as well as the number of households in the district.
> You may want to, and you shall later, compute the average number of rooms/bedrooms/people per household in each district based on the data available to you.

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

```python
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

Basically look at the composition of the thing you're studying (demographic/types of houses/anything else), and purposefully make sure the training set is as representative of reality as possible (by using them fancy sklearn objects), because pure randomness *can* skew heavily one way or another.

Purely random sampling methods are fine if the dataset is large enough (especially in regards to the number of attributes), but if it is not you run the risk of introducing a significant *sampling bias*.

For example if you're sampling a 1000 random people, you need to try to ensure that the Male/Female split is as close as possible to the actual percentages, there's a very high chance that you could be off by a percent or two if sampling purely randomly, that is significant.
If you use purely random sampling, there would be over $10\%$ change of sampling a skewed test set with less than $49\%$ or more than $54\%$ female participants (using the binomial distribution).

> [!INFO]-
> To compute that error probability you can use the binomial distribution:
>
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

![alt text](<../Images/Chapter 2/income_categories.png>)

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

==Since stratified sampling is fairly common, there’s a shorter way to get a single split using the `train_test_split()` function with the stratify argument:==

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

> [!info]- Code for computing the proportions
>
> ```python
> def incom_cate_proportions(data):
>   return data["income_cate].value_counts() / len(strat_test_set)
>
> # Purely random sampling
> train_set, test_set = train_test_split(
>   housing_full, 
>   test_size, 
>   random_state=42)
>
> # Stratified sampling
> strat_train_set, strat_test_set = train_test_split( 
>   housing_full, 
>   test_size=0.2, 
>   stratify=housing_full["income_cat"], 
>   random_state=42) 
>
> compare_props = pd.DataFrame({"
>   Overall %": income_cat_proportions(housing_full),
>   "Stratified %": income_cat_proportions(strat_test_set),
>   "Random %": income_cat_proportions(test_set),
>   }).sort_index()
>
> compare_props.index.name = "Income Category"
>
> compare_props["Strat. Error %"] = (
>   compare_props["Stratified %"] / compare_props["Overall %"] - 1)
>
> compare_props["Rand. Error %"] = (
>   compare_props["Random %"] / compare_props["Overall %"] - 1)
>
> (compare_props * 100).round(2)
> ```

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

![alt text](<../Images/Chapter 2/geographical_scatterplot.png>)

The data is crowded and the points are stacked on top of each other, ==to better visualize high density you can set the alpha option to $0.2$==:

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()
```

![alt text](<../Images/Chapter 2/geographical_scatterplot_alpha.png>)

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

![alt text](<../Images/Chapter 2/geographical_scatterplot_alpha_color_sizes.png>)

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
> ![alt text](<../Images/Chapter 2/california.png>)

## Look for Correlations

### Pearson's r

Since the dataset is not too large, you can easily compute the *standard correlation coefficient* (also called *Pearson’s r*) between every pair of numerical attributes using the `corr()` method:

```python
corr_matrix = housing.corr(numeric_only=True)
```

> [!info]- Pearson correlation coefficient
> From Wikipedia:
>
> In [statistics](https://en.wikipedia.org/wiki/Statistics "Statistics"), the **Pearson correlation coefficient** (**PCC**) is a [correlation coefficient](https://en.wikipedia.org/wiki/Correlation_coefficient "Correlation coefficient") that measures [linear](https://en.wikipedia.org/wiki/Linear "Linear") correlation between two sets of data. It is the ratio between the [covariance](https://en.wikipedia.org/wiki/Covariance "Covariance") of two variables and the product of their [standard deviations](https://en.wikipedia.org/wiki/Standard_deviation "Standard deviation"); thus, it is essentially a normalized measurement of the covariance, such that the result always has a value between −1 and 1.
>
> A key difference is that unlike covariance, this correlation coefficient does not have [units](https://en.wikipedia.org/wiki/Unit_of_measurement "Unit of measurement"), allowing comparison of the strength of the joint association between different pairs of random variables that do not necessarily have the same units.
>
> As with covariance itself, the measure can only reflect a linear [correlation](https://en.wikipedia.org/wiki/Correlation "Correlation") of variables, and ignores many other types of relationships or correlations.
> As a simple example, one would expect the age and height of a sample of children from a school to have a Pearson correlation coefficient significantly greater than 0, but less than 1 (as 1 would represent an unrealistically perfect correlation).

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

![alt text](<../Images/Chapter 2/matrix_scatter_plots.png>)

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

![alt text](<../Images/Chapter 2/income_house_value.png>)

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

> [!Aparté]+
>
> Just throwing spaghetti at the wall here, but since attributes are useful if they tell use *something* about the label, whether the correlation be negative or positive, then we could maybe compute the average value of *r* for the raw attributes and the computed ones, and then compare them to see what's best:
>
> $\langle Raw \rangle = \frac{0.137455+0.054635+|-0.020153|}{3} = 0.070747\overline{6}$
>
> $\langle Computed \rangle = \frac{0.143663+|-0.038224|+|-0.256397|}{3} = 0.146094\overline{6}$
>
> So if we just use the computed values that's a pretty stark improvement, but then again I think you can mix and match attributes and keep only the best, whether raw or computed, idk.

> [!warning]+
> When creating new combined features, make sure they are not too linearly correlated with existing features: collinearity can cause issues with some models, such as linear regression. In particular, avoid simple weighted sums of existing features.

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

Notice how we always keep `strat_train_set` unchanged, **always keep a copy of the original raw data**.

## Clean the data

### Missing features

You need to take care of missing features, for example the `total_bedrooms` attributes has a number of missing values, you have three options to fix this:

1. Get rid of the corresponding entries
2. Get rid of the whole attribute
3. Set the missing values to some value (zero, the mean, the median etc.).
   This is called *imputation*.

You can use the Pandas DataFrame's methods `dropna()`, `drop()` and `fillna()` to do this:

1. - `subset`: specifies which columns to consider for NA checks
   - `inplace` specifies whether to modify the original DataFrame in-place or to return a new one.
2. - `axis`: {0 or ‘index’, 1 or ‘columns’}, default 0 Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
3. - `fillna()` modifies the DataFrame in place

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

> [! Aparté]+
>
> Using this object instead of the pandas methods or even some custom script to apply them is better.
> It saves time, it's consistent, you can apply the same transformation multiple times, you can integrate it into workflows and pipelines with other scikit tools.
> But mostly you don't have to reinvent the wheel, all your spaghetti code will just converge to a shittier version of what scikit already does, so just use that.
>
> Also you **DO NOT** actually wanna change the median as new data arrives, your model was trained on the initial median value, changing it ruins everything.
> Instead of scribbling it on a piece of paper, make the damn scikit object, and export it or save it somewhere or something.
> It doesn't matter if the median does change as new data arrives, and in all cases for a large N and/or a standard distribution the median wouldn't change all that much.
>
> This imputer computes medians once during its 'training' phase (`fit`), and then the values are stored in `statistics_`, you can use them to fill NA values in any of the columns of the data you fed it. As long as you don't retrain it it'll always remember those initial median values and always use them when told to impute.

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
> - **IterativeImputer** trains a regression model per feature to predict the missing values based on all the other available features. It then trains the model again on the updated data, and repeats the process several times, improving the models and the replacement values at each iteration.

#### Scikit learn design - **Read this section**

I can not stress enough how important this section is, also go read the paper linked below.

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
  - *Transformers*: Some estimators can also transform a dataset, these are called `<abbr title="Robots in Disguise">`*transformers*`</abbr>`.
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

Since the output from the transformers is a NumPy array/SciPy sparse matrix, which do not have columns names nor indices, you'll need to wrap it in a DataFrame and recover the columns names and index from `housing_num`:

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

> If you run `sklearn.set_config(transform_output="pandas")`, all transformers will output Pandas DataFrames when they receive a DataFrame as input: Pandas in, Pandas out.

## Handling Text and Categorical Attributes

The only non text attribute in this dataset is `ocean_proximity`, take a look at the first few instances:

```python
>>> housing_cat = housing[["ocean_proximity"]]
>>> housing_cat.head(8)
ocean_proximity
13096 NEAR BAY
14973 <1H OCEAN
3785 INLAND
14689 INLAND
20507 NEAR OCEAN
1286 INLAND
18078 <1H OCEAN
4396 NEAR BAY
```

It's stored as text but you can think of it as categories, there are a limited number of possible values.
Most machine learning algorithms prefer to work with numbers, you'll need to map each category to a number.

### OrdinalEncoder

You can use the Scikit-Learn `OrdinalEncoder` class:

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
# Or call fit() and then transform()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
```

Now those first few values will look something like:

```python
>>> housing_cat_encoded[:8]
array([
[3.], # NEAR BAY
[0.], # <1H OCEAN
[1.], # INLAND
[1.], # INLAND
[4.], # NEAR OCEAN
[1.], # INLAND
[0.], # <1H OCEAN
[3.] # NEAR BAY
])
```

You can get the list of categories using the `categories_instance` variable.
It contains a 1 dimensional array of categories for each categorical attribute (you could have fed it a full DataFrame of all the non numerical attributes, but here you only had one).

```python
>>> ordinal_encoder.categories_
[array(
['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
dtype=object
)]
```

An issue is that ==ML algorithms will assume that two nearby values are more similar than two distant values==, doesn't work for all categorical attributes
(ordered categories like "bad"/"average"/"good" would work, but how can you order country of birth or type of vehicle).

### OneHotEncoder

==A common solution is to create one binary attribute per category, it is equal to 1 if the entry belongs to that category, and 0 if it doesn't.==
If the `ocean_proximity` of some line value is 'ISLAND', then it will have a 1 in the new `ISLAND` column/attribute, and a 0 in every other column which originally came from `ocean_proximity` (`<1 OCEAN` and `INLAND` etc. All the dummy attributes).
This is called *one-hot encoding* because only one attribute will be equal to 1 (hot) while all the others will be 0 (cold).
The new attributes are sometimes called *dummy attributes*.

Use the Scikit-Learn class `OneHotEncoder` to convert categorical values into one-hot vectors:

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

By default, the **output of a OneHotEncoder is a SciPy sparse matrix**, instead of a NumPy array:

```python
>>> housing_cat_1hot
<Compressed Sparse Row sparse matrix of dtype 'float64'
with 16512 stored elements and shape (16512, 5)>
```

Sparce matrices are more efficient at storing matrices that contain mostly zeros, they only keep track of non-zero values and their positions, it saves memory and speeds up computations, however this means that random access is much slower.

When a categorical attribute has a ton of categories, one-hot encoding results in a huge matrix where every row is full of 0s except for 1 value.

You can convert them back to a (dense) NumPy array using the `toarray()` method on the matrix:

```
>>> housing_cat_1hot.toarray()
array([[0., 0., 0., 1., 0.],
[1., 0., 0., 0., 0.],
[0., 1., 0., 0., 0.],
...,
[0., 0., 0., 0., 1.],
[1., 0., 0., 0., 0.],
[0., 0., 0., 0., 1.]], shape=(16512, 5))
```

You could also specify the parameter `sparse_output=False` when **creating** the `OneHotEncoder`, it'll return a regular NumPy array:

```python
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

```python
>>> cat_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
dtype=object)]
```

Pandas has a function called `get_dummies()` which converts each categorical feature into a one-hot representation:

```python
>>> df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
>>> pd.get_dummies(df_test)
	ocean_proximity_INLAND ocean_proximity_NEAR BAY
0   True                   False
1   False                  True
```

Again, same as before, the Scitkit class is superior to doing it 'manually'.
`OneHotEncoder` will remember which categories it was trained on, when your model is in production it should be fed **exactly the same features as during training**.

If we use the `OneHotEncoder` from before on this `df_test` which only contains 2 columns, it'll still output 5 one-hot encoded columns because it considers all the categories it was trained on.
`get_dummies()` only saw 2 categories, so it outputs 2 columns, but the `OneHotEncoder` outputs one column per **learned category**, in the right order.

```python
# The same df_test that only contains "INLAND" and "NEAR BAY", but the one hot encoder remembers all the columns it was trained on and pro
>>> cat_encoder.transform(df_test)
array([[0., 1., 0., 0., 0.],
[0., 0., 0., 1., 0.]])
```

Also if you feed `get_dummies()` a DataFrame containing an unknown category (e.g., "<2H OCEAN") it'll happily generate a column for it, whereas the `OneHotEncoder` will either raise an exception, or if the `handle_unkown` hyperparameter is set to "`ignore`", it'll just represent the unknown category with zeros:

```python
>>> df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
>>> # get_dummies doesn't remember what the categories should be and just generates columns for the new data
>>> pd.get_dummies(df_test_unknown)
	ocean_proximity_<2H OCEAN ocean_proximity_ISLAND
0   True                False
1   False               True
```

```python
>>> cat_encoder.handle_unknown = "ignore"
>>> # the scikit object remembers what the categories are, and if told to ignore new values will just fill those rows with zeros
>>> cat_encoder.transform(df_test_unknown)
array([[0., 0., 0., 0., 0.],
[0., 0., 1., 0., 0.]])
```

> [!Info]+
>
> Using one-hot encoding for categorical attributes with a large number of possible categories can severely slow down training and degrade performance.
> A solution would be to replace the categorical input with **useful numerical features** that are **related to the categories** (e.g., replacing `ocean_proximity` with the distance to the ocean, or a country code with the country's population and GDP per capita).
> You can also one of the encoders provided by the  ![category_encoders</code>package on GitHub](https://github.com/scikit-learn-contrib/category_encoders) .
>
> Or, when dealing with neural networks, you can replace each category with a learnable, low-dimensional vector called an *embedding* (Chapter 14). This is an example of *representation learning* (more examples in Chapter 18).

When you fit *any* Scikit-Learn estimator using a DataFrame, the estimator stores the column names in the `feature_names_in_` attribute.

Scikit-Lean then ==ensures that *any* DataFrame fed to the estimator after that has the same column names== (e.g., to `transform()` or `predict()`).

Transformers also provide the method `get_feature_names_out()` to get back the names of the features, you can use it to build a DataFrame around the transformer's input, it's also useful to avoid column mismatches and when debugging:

```python
>>> cat_encoder.feature_names_in_
array(['ocean_proximity'], dtype=object)
>>> cat_encoder.get_feature_names_out()
array(['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
'ocean_proximity_NEAR OCEAN'], dtype=object)
>>> df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
... columns=cat_encoder.get_feature_names_out(),
... index=df_test_unknown.index)
...
```

## Feature Scaling and Transformation

One of the most important transformations you need to apply to your data is *feature scaling*.

Here the numerical attributes have very different scales, `total_rooms` goes from about 6 to 39,320, while `median_income` only ranges from 0 to 15.
Without any scaling, most models will be biased toward ignoring the median income and focusing on the number of rooms.

There are 2 common ways to get all attributes to have the same scale: *min-max scaling* and *standardization*.

> [!warning]+
>
> Same as all estimators, **only ever use them on the training data**, never use `fit()` or `fit_transform()` for anything else than the training set.
>
> Use the `transform()` method of your trained scaler on any other set, the validation set, the test set, ==any new data== (again you never refit the estimator).
>
> If new data contains outliers, they may end up scaled outside the specified range, if you want to avoid this you can set the `clip` hyperparameter to `True`.

### Min-max scaling

It is also called *normalization*, for each attribute the values are shifted and rescaled so that they end up ranging from 0 to 1.

You simply subtract the min value from all values, then divide the results by the difference between the min and the max.

Scikit-Learn provides a transformer called `MinMaxScaler`, you can change the range using the `feature_range` hyperparameter (e.g., neural networks work best with zero-mean inputs with a range of -1 to 1).

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
```

This type of scaling is very affected by outlying values, suppose a district has a median income equal to 100 (by mistake), instead of the usual 0-15.
Min-max scaling would map this outlier to 1 and crush all the other values down to 0-0.15.

### Standardization

Standardization transforms the values so they follow a normal distribution with $\mu=0$ and $\sigma=1$.

You simply subtract the value from every value then divide them by the standard deviation.

Values aren't restricted to a specific range so it is much less affected by outliers than min-max scaling.

```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

### Heavy tail

When a feature's distribution has a *heavy tail* (i.e., when values far from the mean are **not** exponentially rare), both min-max scaling and standardization will squash most values into a small range.

You'll need to transform the feature to shrink the heavy tail **before** you scale it, you should also try to make the distribution roughly symmetrical.

#### Logs and roots

A common way to do this for **positive features** is to replace the feature with its **square root** (or raise the feature to a power between 0 and 1).

If the feature has a really long and heavy tail, such as a *power law distribution*, you can instead use the logarithm. (plotting the features helps spot this type of thing)

> [!Histograms]-
> ![alt text](<../Images/Chapter 2/histograms.png>)

==For example the *population* feature roughly follows a power law, districts with 10,000 inhabitants are only 10 times less frequent than districts with 1,000 inhabitants, not exponentially less frequent.==

(yes, that huge spike in the beginning is meaningless, to determine whether it follows a power law or not you need to see at which rate do values get less frequent as we move away from the mean, if it's less than exponential then it follows a power law distribution)

Using the log gets this feature much closer to a Gaussian distribution:

![alt text](<../Images/Chapter 2/log_gaussian.png>)

> [!Power law distribution]+
>
> From Wikipedia
>
> In [statistics](https://en.wikipedia.org/wiki/Statistics), a **power law** is a [functional relationship](https://en.wikipedia.org/wiki/Function_\(mathematics\)) "Function (mathematics)") between two quantities, where a [relative change](https://en.wikipedia.org/wiki/Relative_change_and_difference "Relative change and difference") in one quantity results in a relative change in the other quantity proportional to the change raised to a constant [exponent](https://en.wikipedia.org/wiki/Exponent "Exponent"): one quantity varies as a power of another.
> The change is independent of the initial size of those quantities.
> For instance, the area of a square has a power law relationship with the length of its side, since if
> the length is doubled, the area is multiplied by 22, while if the length is tripled, the area is
> multiplied by 32, and so on.
>
> ![alt text](<../Images/Chapter 2/Long_tail.svg>)
>
> An example power-law graph that demonstrates ranking of popularity.
> To the right is the [long tail](https://en.wikipedia.org/wiki/Long_tail "Long tail"), and to the left are the few that dominate (also known as the [80–20 rule](https://en.wikipedia.org/wiki/Pareto_principle "Pareto principle")).

#### Buckets

Another approach for heavy tailed features consists in *bucketizing* the feature.

That is chopping the distribution into roughly equal sized buckets, and replacing each feature value with the index of the bucket it belongs to. (similar to what was done to create the `income_cat` feature, though that was used for stratified sampling)

You could for instance replace each value with its percentile.
Bucketizing with equal-sized buckets results in a feature with an almost uniform distribution, so you don't need any further scaling, or you can divide by the number of buckets to force a 0-1 range.

The point is that the distance between two values is the same, given the values (1, 2, 50, 100), min-max would map them to (0, 0.01, 0.5, 1.0), so the huge gaps to 50 and 100 are preserved, whereas with buckets, if you create 3 equal-sized buckets the values will be (0, 0, 1, 2), you can't tell the original distances between values.

Buckets are useful with heavy tailed distribution *because* they force this uniform spacing, which prevents extreme values from dominating/skewing distributions.

### Multimodal distribution

#### Buckets and categories

Features that have 2 ore more clear peaks, called *modes* are said to have a multimodal distribution, such as the `housing_median_age` feature.

You can bucketize them as well, but this time you need to treat the bucket IDs as categories rather than numerical values.
This means that the bucket indices must be encoded, using a `OneHotEncoder` for example.

This approach will allow the regression model to more easily learn different rules for different ranges of this feature value. For example, perhaps houses built around 35 years ago have a peculiar style that fell out of fashion, and therefore they’re cheaper than their age alone would suggest.

---

 You can think of it as the different modes being sort of like different categories, for example with age, usually you'd think the older the house the cheaper, however if you look at `housing_median_age` plotted against the `median_house_value` you'll see that that's not necessarily the case.

 (here the multimodality in the book was showed by plotting the `housing_median_age` against the number of occurrences with such an age, I'm not sure how that helps us, but I think it might be that it shows the existence of two distinct cohorts of houses, and that they potentially have very different characteristics ergo prices, so they should be treated as categories, idk.)

If you encode the values as usual, then the machine will think a 40 year old house is more similar to 50 year old one than a 20 year old one, or that a 30 year old house is halfway between 10 and a 50 year old houses, which is wrong. The relationship isn't ordinal.
This feature, while numerical, behaves more like a categorical one, and so you should encode it as such.

For features where you expect a 'smooth' relation, like price or population or something, you can still use numerical encoding. (and even then you might be missing something, just plot the data)

![alt text](<../Images/Chapter 2/median_house_value_by_housing_age.png>)

> [!Code]
>
> ```python
> avg_value_by_age = df.groupby('housing_median_age')['median_house_value'].mean()
>
> plt.figure(figsize=(10, 6))
> plt.plot(
>   avg_value_by_age.index, 
>   avg_value_by_age.values, 
>   marker='o', 
>   linewidth=2, 
>   markersize=4
> )
>
> plt.xlabel('Housing Median Age (years)', fontsize=12)
> plt.ylabel('Average Median House Value ($)', fontsize=12)
> plt.title('Average Median House Value by Housing Age', fontsize=14)
> plt.grid(True, alpha=0.3)
> plt.tight_layout()
> plt.show()
>
> ```

![alt text](<../Images/Chapter 2/matrix_scatter_plots.png>)

---

#### RBF

Another approach to transforming multimodal distributions is to ==add a feature for each of the modes== representing the similarity between the housing median age of that entry and that particular mode:

Something like this:

|         | longitude | feature k | housing_median_age | Similarity to mode1 (10 years) | Similarity to mode2 (40 years) |
| ------- | --------- | --------- | ------------------ | ------------------------------ | ------------------------------ |
| Entry i | -120.00   | ...       | 15                 | 0.9                            | 0.05                           |

(Only add a feature for meaningful modes, don't go adding one for those clipping artifacts where every house older than 50 year old gets assigned the value 50, that's no mode)

The similarity measure is typically computed using a *radial basis function* (RBF), that's any function that depends only on the distance between the input value and a fixed point (here one of the modes).

The most commonly used is the Gauss RBF, its output decays exponentially as the input value moves away from the fixed point.

The Gaussian RBF similarity between some housing age x and 35 is given by $\exp(-\gamma(x-35)^2)$.

The hyperparameter $\gamma$ determines how quickly the similarity measure decays as $x$ moves away from 35.

> [!Info]- RBF
>
> From Baeldung
>
> RBF is a mathematical function, say $\psi$, that measures the distance $d$ between an input point (or a vector) $x$ with a given fixed point (or a vector) of interest (a center or reference point) $o$.
> Here, $d$ can be any distance function, such as Euclidean distance. Further, the function $d()$ depends on the specific application and desired set of properties.
>
> $$
> \psi(x) = d(||x-o||)
> $$
>
> As an example, here's a Gaussian RBF, here, the parameter $\gamma$ controls the variance (spread) of the Gaussian curve. A more minor value of $\gamma$ results in a broader curve, while a more considerable value of $\gamma$ leads to a narrower curve:
>
> $$
> \psi(x) = \exp(-\gamma||x-o||)
> $$
>
> ![alt text](<../Images/Chapter 2/gaussian_rbf.png>)

You can use the Scikit-Learn **function** `rbf_kernel()` to create a new Gaussian RBF feature measuring the similarity between the housing median age and 35:

```python
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

The figure below shows what these new feature looks like, as well as what it'd look like with a lower value for gamma. It peaks at 35, the spike in the housing median age distribution, then quickly goes down.

==If this particular age group is well correlated with lower prices, there’s a good chance that this new feature will help.==

![alt text](<../Images/Chapter 2/gaussian_rbf_35.png>)

---

### Transforming target values

The target values ==may== also need to be transformed.

For example if the target distribution has a heavy tail, you may choose to ==replace the target with its logarithm==.
==If you do the regression model will now predict the $\log$ of the median house value, not the median house value itself.==
The Scikit-Learn's transformers have an `inverse_transform()` method to compute the inverse of their transformation.

The example below shows how to scale he labels using a `StandardScaler`, then train a simple linear regression model on the scaled labels, use it to make predictions on new data, then transform the predictions back to the original scale.

> Note that we convert the labels from a Pandas Series to a DataFrame, since the `StandardScaler` expects 2D inputs. Also, in this example we just train the model on a single raw input feature (median income), for simplicity:

```python
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)

some_new_data = housing[["median_income"]].iloc[:5] # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
```

The `TransformedTargetRegressor` is simpler, less error prone and avoids potential scaling mismatches.

First you need to construct it, giving it the regression model and the label transformer.
Then you need to fit it on the training set ==using the original unscaled labels==.
It will automatically use the transformer to scale the labels, then train the regression model on the resulting scaled labels.
Then to make a prediction it will call the regression model's `predict()` method and automatically use the scaler's `inverse_transform()` to produce predictions:

```python
# From before
# housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(
	LinearRegression(),
	transformer=StandardScaler()
)

model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

```

## Custom transformers

You can write your own custom transformations for cleanup operations or combining specific attributes etc.

If your transformation doesn't require any training, you can just write a ==function that takes a NumPy array as input and outputs the transformed array==, then pass the function to Scikit-Learn's `FunctionTransformer`.

Here's an example with logarithm for a positive right-tailed distribution:

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])
```

The `inverse_func` argument is optional.

The transformation function can take hyperparameters as additional arguments, here's a transformer for computing the same Gauss RBF as earlier:

```python
rbf_transformer = FunctionTransformer(
	rbf_kernel,
	kw_args=dict(Y=[[35.]], gamma=0.1)
)
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])
```

There's no inverse for the RBF kernel, since there are always two values at a given distance from a fixed point (except at distance 0 - I think RBF isn't bijective, needs to be bijective to define an inverse function).

Also `rbf_kernel()` doesn't treat features separately, if you pass it an array with n features, it'll compute the n'th norm to measure similarity.
Here's an example that computes the geographic similarity between each district and San Francisco:

```python
sf_coords = 37.7749, -122.41

sf_transformer = FunctionTransformer(
	rbf_kernel,
	kw_args=dict(Y=[sf_coords], gamma=0.1)
)
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])
```

Custom transformers are also useful to combine features. For example, here’s a `FunctionTransformer` that computes the ratio between the input features 0 and 1:
(could use it to compute avg population per house or some such)

```python
>>> ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
>>> ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
array([[0.5 ],
[0.75]])
```

---

### Custom transformer classes

To make your transformer trainable, able to learn parameters using `fit()` and then use its `transform()` method, you'll need to write custom classes.

Here's an example of a custom transformer that groups districts into 10 geographical clusters, then measures the distance between each district and the center of each cluster, adding 10 RBF similarity measures.
(useful if you got 10 cities, proximity to any city tends to increase prices, even more so for some particular ones)

---

Scikit-Learn relies on *duck typing*, "if it looks like a duck and quacks like a duck, it must be a duck", that is to say your object doesn't need to inherit from any base class, it just has to implement a minimum set of functionalities to be considered a transformer.

All you need are three methods:

- `fit()` which **must return `self`**
- `transform()`
- `fit_transform()`, you can get it for free by adding `TransformerMixin` as a base class, the default implementation will just call `fit()` and then `transform`

Also you can get the following methods by adding `BaseEstimator` as a base class and **avoiding** using `*args` and `**kwargs` in the constructor. These methods are useful for automatic hyperparameter tuning:

- `get_params()`
- `set_params()`

Here's a custom transformer that's similar to a `StandardScaler`:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):

	def __init__(self, with_mean=True): # no *args or **kwargs!
		self.with_mean = with_mean

	def fit(self, X, y=None): # y is required even though we don't use it
		X = check_array(X) # checks that X is an array with finite float values
		self.mean_ = X.mean(axis=0)
		self.scale_ = X.std(axis=0)
		self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
		return self # always return self!

	def transform(self, X):
		check_is_fitted(self) # looks for learned attributes (with trailing _)
		X = check_array(X)
		assert self.n_features_in_ == X.shape[1]
		if self.with_mean:
		X = X - self.mean_
		return X / self.scale_

```

> [!Notes]+
>
> - `sklearn.utils.validation` package contains functions to validate inputs.
>   For simplicity they will be skipped in the code of the book.
>   However **production code should have them**.
> - Scikit-Learn pipelines require `fit()` to take two arguments, `X` and `y`, which is why we set `y=None` even though we don't use `y`.
> - All Scikit-Learn estimators set the attribute `n_features_in_` in the `fit()` method, they ensure that the data passed to `transform()` or `predict()` has the same number of features as when the estimator was fitted.
> - `fit()` **must** return `self`.
> - The implementation above is not 100% complete: all estimators should set `feature_names_in` in the `fit()` method.
>   They should also provide a `get_feature_names_out()` method as well as an `inverse_transform()` method if their transformation can be reversed.

A custom transformer can use other estimators in its implementation.
The following code demonstrates a custom transformer that uses a `KMeans` clusterer in `fit()` to identify the main clusters in the training data, then `rbf_kernel()` in `transform()` to measure how similar each sample is to each cluster center:

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):

	def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
		self.n_clusters = n_clusters
		self.gamma = gamma
		self.random_state = random_state

	def fit(self, X, y=None, sample_weight=None):
		self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
		self.kmeans_.fit(X, sample_weight=sample_weight)
		return self # always return self!

	def transform(self, X):
		return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

	def get_feature_names_out(self, names=None):
		return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

> [!KMeans]-
>
> From [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/):
>
> ![alt text](<../Images/Chapter 2/kmeans_gfg_1.webp>)
> ![alt text](<../Images/Chapter 2/kmeans_gfg_2.webp>)
> ![alt text](<../Images/Chapter 2/kmeans_gfg_3.webp>)
> ![alt text](<../Images/Chapter 2/kmeans_gfg_4.webp>)

> [!tldr] API
> You can check whether your custom estimator respects Scikit-Learn’s API by passing an instance to `check_estimator()` from the `sklearn.utils.estimator_checks` package.
> For the full API, check out https://scikit-learn.org/stable/developers.

Here you could use this k-means clustering algorithm to find the most populated regions in California.
The number of clusters is controlled by the `n_clusters` hyperparameter.
The `fit()` method of `KMeans` can take an optional `sample_weight` argument which allows to specify the relative weights of the samples, for example you could pass in the median income so that the clusters are biased towards wealthier districts.
After training, the cluster centers can be accessed via the `cluster_centers_` attribute.

k-means is a *stochastic* algorithm, which means that it relies on randomness to locate the clusters, for reproducible results you should set the `random_state` parameter:

```python
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=67)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
```

All of that code now combined creates a  `ClusterSimilarity` transformer.
You can set the number of clusters to 10, then it call `fit_transform()` with the latitude and longitude of every district in the training set, and uses k-means to locate the cluster centers then the Gaussian RBF similarity between each district and all 10 cluster centers.

The result is a matrix with one row per entry from the data (districts), and one column per cluster:

```python
>>> similarities[:3].round(2)
>>> 
array([
# District1: (SimilarityToCluster1, SimilarityToCluster2, SimilarityToCluster3...)
[0.46, 0. , 0.08, 0. , 0. , 0. , 0. , 0.98, 0. , 0. ],
[0. , 0.96, 0. , 0.03, 0.04, 0. , 0. , 0. , 0.11, 0.35],
[0.34, 0. , 0.45, 0. , 0. , 0. , 0.01, 0.73, 0. , 0. ]]
)
```


![alt text](<../Images/Chapter 2/gaussian_rbf_clusters.png>)

## Transformation Pipelines

There are many data transformation steps that need to be executed in the **right order**.
You can use the Scikit-Learn class `Pipeline` for sequences of transformations, here's an example that can be used to impute then scale numerical attributes:

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
	("impute", SimpleImputer(strategy="median")),
	("standardize", StandardScaler()),
])
```

### Making a pipeline

The pipeline constructor takes a list of tuples of name/estimator pairs which defines a sequence of steps.
==Names are arbitrary as long as they're unique and don't contain double underscores ( __ )==.
They'll be useful for hyperparameter tuning.
==The estimators must all be transformers (they must all have a `fit_transform()` method) except for the last one, which can be anything transformer/predictor/any type of estimator.==


>[!Info]
>
>In a Jupyter notebook, if you import sklearn and run `sklearn.set_config(display="diagram")`, all Scikit-Learn estimators willbe rendered as interactive diagrams. 
>This is particularly useful for visualizing pipelines. 
>To visualize num_pipeline, run a cell with num_pipeline as the last line. 
>Clicking an estimator will show more details.
>
>![[visualize_pipeline.png]]

You can also use the `make_pipeline()` function, it automatically assigns names to transformers based on their classes names (in lowercase and without underscores e.g., `SimpleImputer` -> "`simpleimputer`").
If multiple transformers have the same name, an index is appended to their names (e.g., "`simpleimputer-1`", "`simpleimputer-2`")

### Fit: 
  When you call a pipeline's `fit()` method, it calls `fit_transform()` sequentially on all the transformers, passing the output of each call as the parameter of the next call until it reaches the final estimator for which it just calls `fit()`.
  The pipeline exposes the same methods as the final estimator.
  For example here the pipeline will essentially act the same as a `StandardScaler`.
  
### Transform / Predict: 
  When you call a pipeline's transform method, it will sequentially apply all the transformations to the data.
  ==If the last estimator were a predictor instead of a transformer then the pipeline would have a `predict()` method rather than a `transform()` method.==
  Calling `predict()` would sequentially apply all the transformations to the data and pass the result to the predictor's `predict()` method.

---

Here's an example of that pipeline being used:

```python
>>> housing_num_prepared = num_pipeline.fit_transform(housing_num)
>>> housing_num_prepared[:2].round(2)
array([[-1.42, 1.01, 1.86, 0.31, 1.37, 0.14, 1.39, -0.94],
[ 0.6 , -0.7 , 0.91, -0.31, -0.44, -0.69, -0.37, 1.17]])
```

Then you can recover a DataFrame using the pipeline's `get_feature_names_out()`, the same as any estimator:

```python
df_housing_num_prepared = pd.DataFrame(
	housing_num_prepared, 
	columns=num_pipeline.get_feature_names_out(),
	index=housing_num.index
)
```

### Accessing the estimators

==Pipelines support **indexing**==, `pipeline[1]` returns the second estimator, `pipeline[:-1]` returns ==a `Pipeline` object== containing all but the first estimator.

You can also use the **`steps`** attribute, which is a **list** of name/estimator pairs, or the **`named_steps`** **dictionary** attribute, which maps the names to the estimators.

For example `num_pipeline["simpleimputer"]` returns the estimator named "`simpleimputer`". 

### ColumnTransformer

It's more convenient to have a single transformer capable of handling all columns and applying the appropriate transformation to each, whether they be numerical or categorical.

The following `ColumnTransformer` will apply `num_pipeline` to the numerical attributes and `cat_pipeline` to the categorical attributes (there's just the one in this case):

```python
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
"total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

num_pipeline = make_pipeline(
	SimpleImputer(strategy="median"), 
	StandardScaler()
)
cat_pipeline = make_pipeline(
	SimpleImputer(strategy="most_frequent"),
	OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
	("num", num_pipeline, num_attribs),
	("cat", cat_pipeline, cat_attribs),
])
```

The `ColumnTransformer` takes as an argument to its constructor a list of 3 item tuples.
Each tuple has a string to use as a name for the transformer (must be unique, shouldn't contain double underscores), the transformer itself, then a list of the names of columns on which the transformer should be applied (you can use indices instead of names to refer to the columns, or a `make_column_selector` class as seen below.).

>[!Info]
>
>Instead of using a transformer (in one of those tuples), you can use the string `"drop"` to signify that those columns should be dropped. 
>
>Or the string `"passthrough"` to specify the columns that should be left untouched.
>By default, all the columns that weren't listed will be dropped, but you can set the `remainder` hyperparameter to any transformer (or to `"passthrough"`) for these columns to be handled differently.

#### *`make_column_selector` /  `make_column_transformer()`*

Instead of listing all the column names you can use the Scikit-Learn ==class `make_column_selector`== to ==automatically select all the features of a given type==, such as numerical or categorical.

```python
from sklearn.compose import make_column_transformer

preprocessing = ColumnTransformer([
	("num", num_pipeline, make_column_selector(dtype_include=np.number)),
	("cat", cat_pipeline, make_column_selector(dtype_include=object)),
])
```

If you don't care about naming the transformers you can use the ==function `make_column_transformer()`== (instead of the constructor) which chooses the names for you (just like `make_pipeline()` does):

```python
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
	(num_pipeline, make_column_selector(dtype_include=np.number)),
	(cat_pipeline, make_column_selector(dtype_include=object)),
)

```

You can then apply this `ColumnTransformer` to the housing data:

```python
housing_prepared = preprocessing.fit_transform(housing)
```

>[!warning]+ Return type
>
>Remember how one-hot encoding creates a column for each category which results in a lot of columns with a lot of zeros, so it gets represented as a sparse matrix.
>
>> The *`OneHotEncoder`* returns a *sparse matrix* and the **`num_pipeline`** returns a **dense matrix**.
>> 
>> When there is such a ==mix of sparse and dense matrices==, the `ColumnTransformer` ==estimates the density of the final matrix== (i.e., the ratio of nonzero cells), and it ==returns a sparse matrix if the density is lower than a given threshold== (by default, `sparse_threshold=0.3`). 
>> In this example, it returns a dense matrix.


Now this processing pipeline can take the entire training dataset and apply the appropriate transformer for each column.
It then concatenates the transformed columns horizontally (transformers must **never** change the number of rows).
This returns a NumPy array, which you can wrap in a DataFrame using the column names from `preprocessing.get_feature_names_out()`.


# Recap
This brings everything that was seen up until now together when it comes to preprocessing data.

In the book's example, you'll wanna create a single pipeline that will perform all the transformations, here's what this pipeline will do and why:

- It'll **impute** missing **numerical values** and replace them with the **median**.
  For **categorical features** missing values we'll use the **most frequent category**.
  ==Most ML algorithms don't expect missing values.==
  
- The **categorical feature** will be **one-hot encoded**.
  ==Most ML algorithms only accept numerical inputs.==
  
- A few 'new' **ratio features will be computed and added**: `bedrooms_ratio`, `rooms_per_house`, and `people_per_house`.
  These ==may better correlate with the median house value== and thereby help the ML models.
  
- A few **cluster similarity features will be added**.
  These will likely be more useful to the model than latitude and longitude. (sort of conveys proximity to expensive areas/urban areas and such)
  
- **Features** with a **long tail** will be replaced by their **logarithms**.
  Most ==models prefer== features with roughly ==uniform or Gaussian distributions==.
  
- **All numerical values** will be **standardized**. (standard distribution $\mu=0$ and $\sigma=1$)
  Most ==ML algorithms prefer== when ==**all features** have roughly the **same scale**==.


Here's the code that builds the pipeline to do all of this:

1. Divides every element of the first column by the corresponding element of second column and returns it as a result column (element-wise division, you're supposed to pass it the two columns from which the new feature will be computed)
   Will be used to compute `people_per_house` = `population` / `households` for example.
   `X[:, [n]]` selects all rows of the nth column, the double brackets keep it as a 2D array.
   
2. The `ColumnTransformer` automatically prefixes each feature name with its transformer name (e.g., `"bedrooms" + "__" + "ratio"` -> `"bedrooms__ratio"`)
   For three of the items, the transformers are named after the features not the transformation (`"bedrooms"` rather than `"ratio"`, etc.) , this is to differentiate between them.
   So we use the feature name to convey the transformation applied (`"ratio"`).
   
3. Creates a reusable `Pipeline` for computing generic ratios, the pipeline expects two-column data as an entry, imputes missing values with the median, then it produces a resulting ratio column, scales it and returns it.
   
4. A generic pipeline for transforming a feature into its log, imputes using the median, applies the log (`feature_names="one-to-one"`) preserves the original feature names, then scales it and returns it.
   
5. For each data point, compute similarity to 10 clusters using RBF, then outputs 10 new features, one for each cluster, representing similarity between the datapoint and the cluster.
   
6. A simple pipeline for numerical values, imputes, scales and returns the feature.
   
7. Handles categorical features of any non-numerical type "`dtype_include=object`", imputes with the most common value, ignores values that weren't seen during training.
   
8. THE transformer you'll now use on the raw data. It's a `ColumnTransformer` to process different columns differently at the same time.
   
   For the first three ratio components, each uses `ratio_pipeline()`, each will process those two columns given and output a new feature 
   (e.g., [`"total_bedrooms"`, `"total_rooms"`] -> `"bedrooms__ratio"`)
   
   The `log_pipeline` is applied to almost all numerical values.
   The `cluster_simil` transformer processes latitude and longitude to make a new geographical similarity feature.
   By the end every column of the raw data has been processed save for `"housing_median_age"`, we use the `default_num_pipeline` for this and any remaining feature.
   
   For the names of the resulting features, each is made by concatenating the transformer's name and the original feature's name:
   `final_name` = `transformer_name` + "`__`" + `feature_name`
   For the ratio stuff, the author sort of reversed the transformer's name and feature's name.
   The tuples in column transformer use the feature's name to call the transformer ("`bedrooms"` instead of `"ratio"`, elsewise you'd have 3 transformers all called ratio), and then the `ratio_pipeline()` function uses `"ratio"` as the feature name.

```python
# 1
def column_ratio(X):
	return X[:, [0]] / X[:, [1]]
# 2
def ratio_name(function_transformer, feature_names_in):
	return ["ratio"] # feature names out
# 3
def ratio_pipeline():
	return make_pipeline(
			SimpleImputer(strategy="median"),
			FunctionTransformer(column_ratio, feature_names_out=ratio_name),
			StandardScaler()
		)

# 4
log_pipeline = make_pipeline(
	SimpleImputer(strategy="median"),
	FunctionTransformer(np.log, feature_names_out="one-to-one"),
	StandardScaler()
)

# 5
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# 6
default_num_pipeline = make_pipeline(
	SimpleImputer(strategy="median"),
	StandardScaler()
)

# 7
cat_pipeline = make_pipeline(
	SimpleImputer(strategy="most_frequent"),
	OneHotEncoder(handle_unknown="ignore")
)

# 8
preprocessing = ColumnTransformer(
	[
	("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
	("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
	("people_per_house", ratio_pipeline(), ["population", "households"]),
	("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",                                     "households", "median_income"]),
	("geo", cluster_simil, ["latitude", "longitude"]),
	("cat", cat_pipeline, make_column_selector(dtype_include=object)),
	],
	remainder=default_num_pipeline
	) # one column remaining: housing_median_age
```



The result is a NumPy array with 24 features. 
It include 3 computed ratios, 10 cluster similarity features, 5 categorical features and 6 of the original features that were just transformed.


```python

>>> housing_prepared = preprocessing.fit_transform(housing)
>>> housing_prepared.shape
(16512, 24)
>>> preprocessing.get_feature_names_out()
# All features invariably go through an imputation step before anything else is applied
array([
	# Ratios, notice the naming switcheroo, see above
	'bedrooms__ratio', 
	'rooms_per_house__ratio',
	'people_per_house__ratio', 
	# Numerical attributes, log and standardization
	'log__total_bedrooms',
	'log__total_rooms', 
	'log__population', 
	'log__households',
	'log__median_income', 
	# Clusters, RBF similarity, one-hot encoding
	'geo__Cluster 0 similarity', 
	[...],
	'geo__Cluster 9 similarity', 
	# one-hot encoding
	'cat__ocean_proximity_<1H OCEAN',
	'cat__ocean_proximity_INLAND', 
	'cat__ocean_proximity_ISLAND',
	'cat__ocean_proximity_NEAR BAY', 
	'cat__ocean_proximity_NEAR OCEAN',
	# Standardization
	'remainder__housing_median_age'], 
	dtype=object)

```


# Select and Train a Model

You framed the problem, you got the data, you explored it, sampled a training set and a test set then made a preprocessing pipeline to automatically clean up and prepare the data.
Now you must select and train a machine learning model.

## Train and Evaluate on the Training Set

### LinearRegression

First you can try training a very basic linear regression model to get started:

```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```

And then we can try it out on the training set itself and compare the first five predictions to the labels:

```python
>>> housing_predictions = lin_reg.predict(housing)
>>> housing_predictions[:5].round(-2) # -2 = rounded to the nearest hundred
array([246000., 372700., 135700., 91400., 330900.])
>>> housing_labels.iloc[:5].values
array([458300., 483800., 101700., 96100., 361800.])
```

The first prediction is way off, the others still deviate by around 10% to 25%.

==Remember that you chose to use the RMSE as your performance measure.==
To measure this linear regression model's RMSE on the whole training set, we can use the Scikit-Learn function [`root_mean_squared_error()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html#sklearn.metrics.root_mean_squared_error) (simply pass the true labels and the predicted values to this function):

```python
>>> from sklearn.metrics import root_mean_squared_error

>>> lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
>>> lin_rmse
68972.88910758484
```

We're not using the **`score()`** method to assess performance because it uses a **different metric** than RMSE.
`score()` returns the $R^2$ *coefficient of determination* instead of the RMSE.

This coefficient represents the ratio of the variance in the data that the model can explain, it goes from $-\infty$ to 1, the closer to 1 the better.
If the model simply predicts the mean all the time and does not explain any of the variance, then $R^2=0$. If the model does even worse than that, then the score can be negative.
Here's a helpful [StatQuest video](https://www.youtube.com/watch?v=2AQKmw14mHM) on $R^2$.

Also a reminder regarding RMSE:

![[Chapter 2#RMSE ($l_{2}$ norm)]]

---

Most median values range from $\$120,000$ to $\$265,000$, a typical prediction error of almost $\$69,000$ is unacceptable.
This is an example of **underfitting**. 

![[Studying/Self/HOMLP/Notes/Chapter 1#Underfitting the training data]]

This can mean that the features do not provide enough information to make good predictions, or that the model is not powerful enough.
The model is not regularized so there are no constraints to be reduced, and before trying to add more features you could try a more complex model.

### DecisionTreeRegressor
This is a fairly powerful model capable of finding complex nonlinear relationships in the data. (decision trees will be presented in more detail in Chapter 5):

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
```

And then you may evaluate it on the training set:

```python
>>> housing_predictions = tree_reg.predict(housing)
>>> tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
>>> tree_rmse

0.0
```

When you see no error at all, it's much more likely that the model has badly overfit the data, rather than it being perfect.
You should **not** touch the test set until you are ready to launch a model you are very confident about.
So in order to assess whether this model is good or has overfitted, you'll need to ==use part of the training set for training, and part of it for model validation==.

Such an evaluation isn't utterly useless though, a seemingly great performance here with a poor one in a different evaluation technique with validation sets may indicate overfitting.

## Better Evaluation Using **Cross-Validation**
You could simply use the function `train_test_split()` once more to split the training set into a smaller training set and a validation set, then train models against the smaller training set and evaluate them on the validation set. It's a bit cumbersome but it works.

A great alternative is to use Scikit-Learn's *k-fold cross-validation* feature.

It splits the training set into **k non-overlapping subsets called folds**, then the model is trained and evaluated $k$ times.
For each iteration $i$ from 1 to $k$, the model is trained the whole dataset save for fold number $i$, then it is evaluated against the $i^{th}$ fold, and so on.
This process produces $k$ evaluation scores.

![[k-fold_cross-validation.png]]

Scikit-Learn provides a [`cross_val_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) function which can take in as arguments the estimator, the data, the labels, a scoring strategy to evaluate the performance of the estimator across cross-validation splits and the number of folds:

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(
	tree_reg, 
	housing, 
	housing_labels,
	scoring="neg_root_mean_squared_error", 
	cv=10
)
```

>[!Danger] Minus sign
>
>Scikit-Learn's cross-validation functions such as `cross_val_score` expect a utility function, a higher score means the performance is better. (e.g., accuracy, $R^2$, precision)
>Whereas for error metrics lower is better. (e.g., MSE, RMSE, MAE).
>
>To unify the logic across the board, Scikit-Learn returns the negative values for metrics that need to be lowered, including the RMSE, so that when you're selecting the best
>
>Pseudocode to illustrate, here's what sklearn would need to do without this convention:
>```python
>if metric in ["accuracy", "r2", "precision"]: 
>	best_model = model_with_HIGHEST_score 
>elif metric in ["rmse", "mae", "mse"]: 
>	best_model = model_with_LOWEST_score 
>else: 
>	???
>```
>
>Now with that convention, where you flip the sign of what needs to be minimized, you always just gotta look for the highest score:
>
>```python
># Across any and all metrics
>best_model = model_with_HIGHEST_score 
>```

The values returned by the function are negative because of the API implementation detail explained above, just flip the sign then take a look at the results:

```python
>>> pd.Series(tree_rmses).describe()
count 10.000000
mean 66573.734600
std 1103.402323
min 64607.896046
25% 66204.731788
50% 66388.272499
75% 66826.257468
max 68532.210664
dtype: float64
```

Now you can see that the decision tree isn't performing well at all, it looks almost as bad as the linear regression model.

Notice that ==cross-validation allows you to get an estimate of the performance of the model **and** how precise this estimate is (its standard deviation)==.
Here the RMSE is about $\$66,574$ and $\sigma=\$1103$.
The only downside is that the model needs to be trained several times, which can be costly and isn't always feasible.

Computing the same metric on the linear regression model yields a RMSE of $\$70,003$ and $\sigma=\$4182$.
The decision tree seems to perform slightly better, but the difference is minimal due to sever overfitting.

**We know there's an overfitting problem because the training error (actually zero) is low while the validation error is high.**

---

### RandomForestRegressor

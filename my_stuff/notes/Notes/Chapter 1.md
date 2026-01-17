
![Overview of Machine Learning Categories](ml-categories.png)

# Definition

>Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.

# Use-cases

Machine learning is great for:

- Problems for which existing solutions require a lot of work and maintenance, such as long lists of rules.
- Complex programs for which using a traditional approach yields no good solution.
- Fluctuating environments (a machine learning system can easily be retrained on new data, always keeping it up to date)
- Getting insights about complex problems and large amounts of data (data mining)

# Examples of applications


| Example                                                                                                            | Type of problem             | Techniques that can be used                                                                                                                                                                |
| :----------------------------------------------------------------------------------------------------------------- | :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Analyzing images of products on a production line to automatically classify them                                   | Image classification        | CNNs / Vision transformers                                                                                                                                                                 |
| Detecting tumors in brain scans                                                                                    | Semantic image segmentation | CNNs / Vision transformers                                                                                                                                                                 |
| Automatically classifying news articles                                                                            | NLP (text classification)   | RNNs / CNNs / transformers                                                                                                                                                                 |
| Automatically flagging comments on discussion forumes                                                              | NLP (text classification)   | RNNs / CNNs / transformers                                                                                                                                                                 |
| Summarizing long documents automatically                                                                           | NLP (text summarization)    | RNNs / CNNs / transformers                                                                                                                                                                 |
| Estimating a person's genetic risk for a given disease by analyzing a very long DNA sequence                       | State Space Models (SSMs)   | State Space Models (SSMs)                                                                                                                                                                  |
| Creating a chatbot or a personal assistant                                                                         | NLP (various components)    | Natural Language Understanding (NLU), question-answering modules...                                                                                                                        |
| Forecasting your company’s revenue next year, based on many performance metrics                                    | Regression                  | Linear regression/Polynomial regression/regression support vector machine/regression random forest...<br>RNNs/CNNs/transformers to take into account sequences of past performance metrics |
| Making your app react to voice commands                                                                            | Speech recognition          | RNNs/CNNs/transformers                                                                                                                                                                     |
| Detecting credit card fraud                                                                                        | Anomaly detection           | Isolation forests/Gaussian mixture models/autoencoders                                                                                                                                     |
| Segmenting clients based on their purchases so that you can design a different marketing strategy for each segment | Clustering                  | k-means/DBSCAN...                                                                                                                                                                          |
| Representing a complex, high-dimensional dataset in a clear and insightful diagram                                 | Data visualization          | Dimensionality reduction techniques                                                                                                                                                        |
| Recommending a product that a client may be interested in, based on past purchases                                 | Recommender system          | Artificial neural network                                                                                                                                                                  |
| Building an intelligent bot for a game                                                                             | Reinforcement learning (RL) | RL                                                                                                                                                                                         |

One thing I'm seeing is that RNNs/CNNs/transformers are for long sequences of data that's usually organized in some way (e.g. in text word2 comes after word1, or in speech syllable2 comes after syllable1).

# Types of machine learning systems

## Supervised vs unsupervised

### Supervised

Supervised means the data fed to the algorithm (training data) includes the desired solution, each entry is tagged/labeled. 

e.g. for a regression task where we aim to predict the price (the *target*) of a car given some specs (called *features*), the  system would be trained on data containing both the features (mileage, age, brand...) and the targets (each car's price).

> The words *target* and *label* are generally treated as synonyms in supervised learning, but *target* is more common in **regression** tasks and *label* is more common in **classification** tasks.
> Moreover, features are sometimes called predictors or attributes. These terms may refer to individual samples (e.g., “this car’s mileage feature is equal to 15,000”) or to all samples (e.g., “the mileage feature is strongly correlated with price”).

(Makes sense, for regression you get the raw data and shoot for a predicted value, for classification you got ready-made categories and are trying to see in which does the data fit.)

### Unsupervised

The training data is unlabeled.

Examples are clustering algorithms that tries to group the data together into *clusters* of entries with a high degree of similarity.

Visualization algorithms can be used to turn a lot of complex and unlabeled data into a 2D/3D representation of the data.

Anomaly detection, the system is shown mostly normal instances during training and learns to recognize them. When it sees a new instance it can tell whether it's a normal one or likely an anomaly. A similar task is novelty detection to detect new instances that look different from all instances in the training set.

Association rule learning can be used to discover interesting relations between attributes.

A related task is dimensionality reduction, in which the goal is to simplify the data without losing too much information. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be strongly correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called feature extraction.

>It is often a good idea to try to reduce the number of dimensions in your training data using a dimensionality reduction algorithm before you feed it to another machine learning algorithm (such as a supervised learning algorithm). It will run much faster, the data will take up less disk and memory space, and in some cases it may also perform better.


### Semi-supervised learning

Training data is partially labeled, useful because labeling data is very time-consuming and costly.

Usually a combination of unsupervised and supervised algorithms, for instance a clustering algorithm to group similar instances together, and then the unlabeled instances can be labeled with the most common label in its cluster.

### Self-supervised learning

Aims to generate a fully labeled dataset from a fully unlabeled one. Once that's done any supervised learning algorithm can be used.

For example, if you have a large dataset of unlabeled images, you can randomly mask a small part of each image and then train a model to recover the original image. During training, the masked images are used as the inputs to the model, and the original images are used as the labels. 
Such a model may be useful to repair damaged images or erase unwanted objects from pictures. 

Often they're tweaked to be used for a different task, for instance for the model that repairs images, it may be given different pictures of different animals and trained to repair them, repairing an image correctly entails 'knowing' what sort of animal is on it, the model can be tweaked to instead be used to categorize animals.

LLMs are trained in a similar way, by masking random words in a huge text corpus and training the model to predict the missing words.

> Transferring knowledge from one task to another is called transfer learning, and it’s one of the most important techniques in machine learning today, especially when using deep neural networks (i.e., neural networks composed of many layers of neurons). 
> 
> (Discussed in detail in Part II).


### Reinforcement learning

The learning system is called an *agent*, it observes its environment, selects and performs actions, and gets *rewards* or *penalties* in return.
It must learn by itself what the best *policy* to get the most reward over time is.


## Batch versus online learning

Batch learning means training the model on the full data set.
Online learning means training it on a batch of data at a time, for example using *gradient descent*.

### Batch learning (offline learning)

The system is rained using all the available data, it is a costly and time-consuming task, and so it is typically done offline.
First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.

The model's performance tends to decay over time, simply because the world continues to evolve while the model remains unchanged. 

This phenomenon is often called *data drift* (or *model rot*).

The model should be regularly retrained on new data. How often you need to do that depends on the use case: if the model classifies pictures of cats and dogs, its performance will decay very slowly, but if the model deals with fast-evolving systems, for example making predictions on
the financial market, then it is likely to decay quite fast.

If new data appears, for instance a new type of spam, the model must be retrained on the full dataset, including both the old data and the new.

### Online learning (incremental learning)

The system is trained incrementally by feeding it data instances sequentially. Either individually or in small groups called *mini-batches*.
Each learning step is fast and cheap, the system can learn about new data as it arrives.
The most common online algorithm is *gradient descent*.

This is useful for systems that need to adapt to change extremely rapidly, or if computing resources are limited. Online learning can also be used to train models on huge datasets that cannot fit in one machine's memory (*out-of-core* learning).

The *learning rate* is how fast the system should adapt to changing data. 
With a high learning rate the system adapts quickly to new data but tends to forget the old data, this is called *catastrophic forgetting* or *catastrophic interference*.
With a low learning rate the system will learn more slowly, but will be less sensitive to noise or outliers in the new data.

Online learning is very sensitive to bad data, for example bad data coming from a bug or malicious activity. You need to monitor the system closely and switch learning off or revert to a previously working state if a drop in performanceis detected.

## Instance-based versus model-based learning

This categorizes machine learning systems by how they *generalize*. 
That means making good predictions on never-seen before examples based on previous training.

If the algorithm is model-based, it tunes some parameters to fit the model to the training set (i.e., to make good predictions on the training set itself), and then hopefully it will be able to make good predictions on new cases as well. 

If the algorithm is instance-based, it just learns the examples by heart and generalizes to new instances by using a similarity measure to compare
them to the learned instances.
### Instance-based learning


Based on a measure of similarity with previous examples that have been learned by heart.

Given a new instance, the system compares it to learned examples (or a subset of them) and tries to classify it.

>Instance-based learning often shines with small datasets, especially if the data keeps changing, but it does not scale very well: it requires deploying a whole copy of the training set to production; making predictions requires searching for similar instances, which can be quite slow; and it doesn’t work well with high-dimensional data such as images.

![alt text](instance-based_learning.png)

### Model-based learning and a typical machine learning workflow
Builds a model based on a set of examples and uses it to make *predicitions*.

Essentially a model is a mathematical function, any new entry would have to fit the model (the function or equation), and so we can use that 



![Model-based learning](model-based_learning.png)

Each model has a number of *model parameters*, based on the type of the equation.

For instance if we plot life satisfaction and GDP per capita, we can see that they evolve more or less linearly (barring some noise), so we could use a linear model (here with two parameters):

![Linear model](linear-model.png)

$$
life\_satisfaction = \theta_{0} + \theta_{1}*GDP\_per\_capita
$$
Tweak $\theta_0$  and $\theta_{1}$ until the function fits the data, that's your model.
To know which values make the model perform best, you'll need to specify a performance measure, either:
- A *utility function*/*fitness function* that measures how *good* the model is.
- Or a **cost function**/**loss function** that measures how **bad** it is.

For linear regression problems people typically use a **cost** function that measures the distance between the linear model's predictions and the training examples, the objective is to minimize this distance.

![Linear model](linear-model-best-fit.png)


# Typical workflow

- Study the data
- Select a model
- Train the model on the training data (i.e., the learning algorithm searched for the model parameter values that minimize a cost function)
- Apply the model to make predictions on new cases (*inference*)

# Main challenges of machine learning
The two main things that can go wrong are "bad model" and "bad data".

## Bad data
### Insufficient quantity of training data

Machine learning requires a lot of data for most machine learning algorithms to work properly. 

![The importance of data versus algorithms](data-over-algorithms.png)

> In a famous paper published in 2001, Microsoft researchers Michele Banko and Eric Brill showed that very different machine learning algorithms, including fairly simple ones, performed almost identically well on a complex problem of natural language disambiguation6 once they were given enough data (as you can see in Figure 1-21).
> 
> As the authors put it, “these results suggest that we may want to reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development”. 
> The idea that data matters more than algorithms for complex problems was further popularized by Peter Norvig et al. in a paper titled “The Unreasonable Effectiveness of Data”, published in 2009.
> 
> **It should be noted, however, that small and medium-sized datasets are still very common, and it is not always easy or cheap to get extra training data, so don’t abandon algorithms just yet.**

### Nonrepresentative training data

Same problem as statistics.
If the sample is too small, you will have *sampling noise* (i.e. nonrepresentative data as a result of chance).
A very large sample can also be nonrepresentative if the sampling method is flawed (*sampling bias*).

### Poor quality data

That is data full of errors, outliers and noise.

==It is well worth the effort to spend time cleaning up the training data, most data scientists spend a significant part of their time doing just that.==


The following are a couple examples of when you’d want to clean up training data:

• If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually.
• If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it.

### Irrelevant Features
The system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones.

==A critical part of the success of a machine learning project is coming up with a good set of features to train on.==

This process, called *feature engineering*, involves the following steps:

- *Feature selection* (selecting the most useful features to train on among existing features)
- *Feature extraction* (combining existing features to produce a more useful, may use **dimensionality reduction** algorithms)
- Creating new features by gathering new data

## Bad algorithms

### Overfitting the training data

**Over**generalizing = **Over**fitting = Model is **overly** complex

The model performs well on the training data, but it does not generalize well. 

That is to say it thinks all data out there must look like the training data, it **overgeneralizes** based on its limited experience, the model fits too closely to the training set and can't be applied to real life.

---

If the training set is noisy, or if it is too small, which introduces sampling noise, then the model is likely to detect patterns in the noise itself. 
Said patterns will not generalize to new instances.

>For example, say you feed your life satisfaction model many more attributes, including uninformative ones such as the country’s name. In that case, a complex model may detect patterns like the fact that all countries in the training data with a w in their name have a life satisfaction greater than 7: New Zealand (7.3), Norway (7.6), Sweden (7.3), and Switzerland (7.5). How confident are you that the w-satisfaction rule generalizes to Rwanda or Zimbabwe?

![The importance of data versus algorithms](overfitting.png)

---

Overfitting happens when the model is ==too complex relative to the amount and noisiness of the training data==, so it starts to learn random patterns in the training data. Here are possible solutions:

• Simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training
data, or by constraining the model.
• Gather more training data.
• Reduce the noise in the training data (e.g., fix data errors and remove outliers).

---

Constraining a model to make it simpler and reduce the risk of overfitting is called *regularization*.

This involves playing around with the *degree of freedom* of the algorithm, for instance in the linear model seen before:

$$
life\_satisfaction = \theta_{0} + \theta_{1}*GDP\_per\_capita
$$
There are two degrees of freedom, the height ($\theta_0$) and the slope ($\theta_1$).

If we forced $\theta_1 = 0$, the algorithm would have only one degree of freedom and would have a much harder time fitting the data, it would simply move up and down and eventually end up around the mean.

==If we allow the algorithm to modify $\theta_1$ but keep it small, then the learning algorithm will effectively have somewhere in **between one and two degrees of freedom**. ==

It will produce a model that’s simpler than one with two degrees of freedom, but more complex than one with just one.

---

Regularization can be used to reduce the risk of overfitting, the amount of regularization to apply during learning can be controlled by a *hyperparameter*.

A hyperparameter is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training. 

If you set the regularization hyperparameter to a very large value, you will get an almost flat model (a slope close to zero); the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution.

### Underfitting the training data

*Under*fitting = Model is *too simple*

Occurs when the model is too simple to learn the underlying structure of the data, when reality is just more complex than the model picked.

Here are the main options for fixing this problem:

• Select a more powerful model, with more parameters.
• Feed better features to the learning algorithm (feature engineering).
• Reduce the constraints on the model (for example by reducing the regularization hyperparameter).

### Deployment issues
e.g. the model is too complex to maintain, too large to fit in memory, too slow, doesn't scale properly, has security vulnerabilities, becomes outdated quickly etc.

Different skillset from that of data modelling, this is the domain of *MLOps* (ML operations).


# Testing and Validating

Data is split into two sets, the *training set* and the *test set*.

The error rate on the new cases is called the *generalization error* or *out-of-sample error*, it is estimated by evaluating the model on the test set. This values represents how well the model will perform on instances it has never seen before.

**If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the training data.**

>It is common to use 80% of the data for training and hold out 20% for testing. However, this depends on the size of the dataset: if it contains 10 million instances, then holding out 1% means your test set will contain 100,000 instances, probably more than enough to get a good estimate of the generalization error.

# Hyperparameter Tuning and Model Selection
One option to choose between two types of models (a linear model and a polynomial model for example) is to train both and compare how well they generalize using the test set (it must be the same test set I believe).

Now suppose the linear model generalizes better, and we want to apply some regularization to avoid overfitting. 

One option to choose the value of the regularization parameter is to train 100 different models using 100 different values for this hyperparameter, then pick the best.

However this model wouldn't perform as well as expected.
==The problem is that you *measured the generalization error multiple times on the test set*, and you *adapted the model and hyperparameters* to produce the best model *for that particular set*. 
This means the model is *unlikely to perform as well on new data*.==

A common solution to this problem is called *holdout validation*: you simply hold out part of the training set to evaluate several candidate models and select the best one. The new held-out set is called the *validation set* (or the *development set*, or *dev set*). 

More specifically, you train multiple models with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set), and you select the model that performs best on the validation set.
After this holdout validation process, you train the best model on the full training set (including the validation set), and this gives you the final model. Lastly, you evaluate this final model on the test set to get an estimate of the generalization error.[^1]

![](holdout-validation.png)

The validation set shouldn't be too small, as the model evaluations will be imprecise. 
And it shouldn't be too large, so that the full training set isn't much larger than the remaining training set. 
The final model will be trained on the full training set, it is not ideal to compare candidate models trained on a much smaller training set, they might perform well on that, but not on the full data.

One way to solve this problem is to perform repeated *cross-validation*, using many small validation sets. 
Each model is evaluated once per validation set after it is trained on the rest of the data. 
By averaging out all the evaluations of a model, you get a much more accurate measure of its performance.  Time-costly though.



[^1]: This seems to me like some sort of partial testing, you first train all of them models with the different hyperparameters on part of the data. Then you evaluate which hyperparameter value is the best with that validation step. Then you evaluate how well the whole thing performs with that final test.

# Data mismatch

If you can get large amounts of data for training, but this data is not perfectly representative of the data that will be used in production (e.g. your app only has 1000 users and so 1000 entries, but there are full datasets available online of mostly similar apps).

**In this case, the most important rule to remember is that both the validation set and the test set must be as representative as possible of the data you expect to use in production, they should be composed exclusively of representative pictures.**
You can shuffle them and put half in the validation set and half in the test set (making sure that no duplicates or near-duplicates end up in both sets).

You can train your model on the less-than-ideal-but-plentiful data, and then check its performance on the validation set, however if the performance is disappointing, **you will not know whether this is because your model has overfit the training set, or whether this is just due to the mismatch between the training data and the real life data.**

One solution is to hold out some of the training data in a *train-dev set*.
After the model is trained it can be evaluated on this *train-dev set*, if it performs poorly that means it overfit the training set, so it must be simplified, regularized or we must get more training data or clean it.

If it performs well on the *train-dev set* we can then evaluate it on the *validation set*.
If it perform poorly on the *validation set* then the problem must be coming from the data mismatch, you can try to preprocess the training data to make it more similar to real life stuff 

(e.g. the training set contains stock images and the real life data is about smartphone pictures, may reduce the quality or resolution of the stock images to approximate those of a phone, or if most phone pictures are in portrait mode then only keep those in the training set...)

Once you have a model that performs well on both the train-dev set and the dev set, you can evaluate it one last time on the test set to know how well it is likely to perform in production.

- *training set*: Raw data entries, just feed the model and let it learn. (e.g. stock plant images)
- *train-dev set*: Check if the model can make predictions on data of the **very same type** as the training set, same thing that it's been trained on. (stock plant images as well)
- *validation set*/*dev set*: Check if the model can make predictions on the type of data we're actually interested in, in order to see which hyperparameters are best. (e.g. smartphone pictures of plants)
- *test set*: Check if the final model with all the hyperparameters we picked and everything performs well on the type of data we're actually interested in (smartphone pictures of plants).

![](solution-to-scarce-data.png)














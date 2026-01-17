The Root mean squared error (RMSE):

$$
RMSE(\mathbf{X}, \mathbf{y}, h) = \sqrt{ \frac{1}{m} \sum_{i=1}^m ( h( \mathbf{x}^{(i)} ) - y^{(i)} )^2 }
$$

This equation introduces several very common machine learning notations that I will
use throughout this book:

- $m$ is the **number of instances in the dataset** you are measuring the RMSE on.
  For example, if you are evaluating the RMSE on a validation set of 2,000 districts, then m = 2,000.
- $\mathbf{x}^{(i)}$ is a **vector** of all the **feature values** (excluding the label/tag/solution) of the $i^{th}$ instance in the dataset, and $y^{(i)}$ is its label (the desired output value for that instance). $y$ is a vector containing the labels of all the instances in the dataset.

  For example, if the first district in the dataset is located at longitude –118.29°, latitude 33.91°, and it has 1,416 inhabitants with a median income of $38,372, and the median house value is $156,400 (ignoring other features for now), then:

$$
X^{(1)} = \begin{pmatrix}
-118.29\\
33.91 \\
1,416\\
38,372
\end{pmatrix}
$$

and:

$$
y^{(1)} = 156,400
$$

> $\mathbf{X}$ is all the inputs or features, $y$ is all the desired outputs or tags.
> $\mathbf{x}^{(i)}$ is input number i, the features of entry number i in the dataset, $y^{(i)}$ is the label associated with those features, the label of entry number i.

- $\mathbf{X}$ is a matrix containing all the feature values (excluding labels) of all instances in the dataset. There is one row per instance, and the $i^{(th)}$ row is equal to the transpose of $\mathbf{x}^{{(i)}}$, denoted $(\mathbf{x}^{{(i)}})^\intercal$.
  For example if the district is as just described, then the matrix $X$ looks like this:

$$
\mathbf{X} = \begin{pmatrix}
(\mathbf{x}^{{(1)}})^\intercal \\
(\mathbf{x}^{{(2)}})^\intercal \\
\vdots \\
(\mathbf{x}^{{(1999)}})^\intercal \\
(\mathbf{x}^{{(2000)}})^\intercal
\end{pmatrix}
= 
\begin{pmatrix}
\\ \\
\\ \\
-118.29 & 33.91 & 1,416 & 38,372 \\
\vdots & \vdots & \vdots & \vdots \\ \\
\\ \\

\end{pmatrix}
$$

> Don't let this shit scare you, it just says that each entry is a vector of features, *vertical* vector, and then to combine them into X you flip each of them and stack them, so that each line is an entry, and each column of X represents a feature.

> Basically the data is structured like this:
>
> |                     | Feature 1 | Feature 2 | Feature n |
> | ------------------- | --------- | --------- | --------- |
> | Entry 1 (transpose) | x1_1      | x1_2      | x1_n      |
> | Entry 2 (transpose) | x2_1      | x2_2      | x2_n      |
> | Entry n (transpose) | xn_1      | xn_2      | xn_n      |

> With each entry being a vector:
>
> |           | Entry 1 |
> | --------- | ------- |
> | Feature 1 | x1_1    |
> | Feature 2 | x1_2    |
> | Feature n | x1_n    |

- $h$ is your system's prediction function, also called a *hypothesis*.
  When your system is given an instance's feature vector $\mathbf{X}^{(i)}$, it ouputs a predicted value $\hat{y}^{(i)} = h(\mathbf{X}^{(i)})$ for that instance.
  For example, if your system predicts that the median housing price in the first district is $\$158,400$, then $\hat{y}^{(1)} = h(\mathbf{X}^{(1)}) = 158,400$. 
  
  The prediction error for this district is $\hat{y}^{(1)} - y^{(1)} = 158,400 - 156,400 = 2000$
- $RMSE(\mathbf{X}, y, h)$ is the cost function measured on the set of examples using your hypothesis $h$.

We use lowercase italic font for scalar values (such as $m$ or $y^{(i)}$) and function names (such as $h$), lowercase bold font for vectors (such as $\mathbf{x}^{(i)}$), and uppercase bold font for matrices (such as $\mathbf{X}$).

# ML-Algorithms
Machine Learning (ML) is a field of artificial intelligence (AI) that focuses on developing algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data. It is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on **patterns** and **inference** instead.

### Types of Machine Learning
**Supervised Learning:** It involves training a model on a **`labeled dataset`**, where each example is paired with the correct answer. The model learns to **`map inputs to outputs`**, making predictions on new data.

**Unsupervised Learning:** Here, the model is trained on **`unlabeled data`**, and it learns to find patterns or structures in the data without explicit guidance. Clustering and dimensionality reduction are common tasks in unsupervised learning.

**Semi-supervised Learning:** This is a combination of supervised and unsupervised learning, where the model is trained on a dataset that contains both labeled and unlabeled data.

**Self-supervised Learning:** A type of ML where a model learns to predict certain parts or properties of its input data without human-provided labels. Instead, the model generates its own supervision signal from the input data. This approach is often used in situations where labeled data is scarce or expensive to obtain.

**Reinforcement Learning:** This type of ML involves an **`agent`** learning to make decisions by interacting with an **`environment`**. The agent receives **`feedback`** in the form of **`rewards`** or **`penalties`**, which guides its learning process.

**Deep Learning:** Deep learning is a subset of ML that uses **`neural networks`** with many layers (deep neural networks) to learn complex patterns in large amounts of data. It has been particularly successful in tasks like image recognition and natural language processing.

## Machine Learning Models
**Linear Regression**  : For this model I will first split the dependent and independent variables using slicing method of Python.
Then I will make a function to update the weights, and two more functions to differentiate the cost function with respect to w1 and w0.
Then I will make a class of Linear Regression, in this class there will be a constructor, predict function, update function and 
gradient descent function.

**Logistic Regression** :  For this model I will first split the dependent and independent variables using slicing method of Python.
Then I will make a Sigmoid function so that all values lies between 0 to 1, and a Cost function. Then I will make a class of
Logistic Regression in which there will be a constructor, predict function, update function and gradient descent function.


**KNN(K Nearest Neighbours)** : In this model if I want to classify a datapoint then I have to compare it with it’s neighbouring datapoints.
For this first I will split the dependent and independent variables using slicing method of Python. Then I will make an Euclidean 
function to measure the distance between two datapoints. Then I will make a class which will contain a constructor and a KNN function.
In KNN function I will first measure the euclidean distance between the input and training datapoints and then I will sort it in ascending order.
Then if we iterate it over the first K elements, we will get the index of the most occurring value.


**K-Means Clustering** : For this model first I will make an Euclidean function to measure the distance between two datapoints.
Then I will make a class for KMean which consist of a constructor and a function which will first randomly initialize the clusters centers,
then after evaluation it will give a list of cluster centers closer to each datapoint. 


**Decision Tree** : A Decision Tree is a tree-like model used for classification and regression. It splits the data into smaller subsets
based on features, making decisions at each node to reach a prediction at the leaf nodes. They're easy to interpret but can overfit with complex data.

**Random Forest** : Random Forest is an ensemble learning method that uses multiple Decision Trees to improve predictions. It builds several trees
and merges them to reduce overfitting and improve accuracy. It's robust, handles large datasets well, and provides feature importance estimates.

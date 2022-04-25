# (PART) Machine Learning Courses {.unnumbered}

<h3 class="course__description-title">Course Description</h3>
<p class="course__description">Machine learning is the field that teaches machines and computers to learn from existing data to make predictions on new data: Will a tumor be benign or malignant? Which of your customers will take their business elsewhere? Is a particular email spam? In this course, you'll learn how to use Python to perform supervised learning, an essential component of machine learning. You'll learn how to build predictive models, tune their parameters, and determine how well they will perform with unseen data—all while using real world datasets. You'll be using scikit-learn, one of the most popular and user-friendly machine learning libraries for Python.</p>

# Classification

<p class="chapter__description">
    In this chapter, you will be introduced to classification problems and learn how to solve them using supervised learning techniques. And you’ll apply what you learn to a political dataset, where you classify the party affiliation of United States congressmen based on their voting records.
  </p>
  
## Supervised learning

### Which of these is a classification problem?

<div class=""><p>Once you decide to leverage supervised machine learning to solve a new problem, you need to identify whether your problem
is better suited to classification or regression. This exercise will help you develop your intuition for distinguishing between the two.</p>
<p>Provided below are 4 example applications of machine learning. Which of them is a supervised classification problem?</p></div>

<ul>
<strong><li><div class="">Using labeled financial data to predict whether the value of a stock will go up or go down next week.</div></li></strong>
<li><div class="">Using labeled housing price data to predict the price of a new house based on various features.</div></li>
<li><div class="">Using unlabeled data to cluster the students of an online education company into different categories based on their learning styles.</div></li>
<li><div class="">Using labeled financial data to predict what the value of a stock will be next week.</div></li>
</ul>

### Exploratory data analysis



### Numerical EDA


<div class>
<p>In this chapter, you'll be working with a dataset obtained from the <a href="https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records">UCI Machine Learning Repository</a> consisting of votes made by US House of Representatives Congressmen. Your goal will be to predict their party affiliation ('Democrat' or 'Republican') based on how they voted on certain key issues. Here, it's worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models. Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself - including on this very same dataset!</p>
<p>Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data. For a refresher on the importance of EDA, check out the first two chapters of <a href="https://www.datacamp.com/courses/statistical-thinking-in-python-part-1">Statistical Thinking in Python (Part 1)</a>. </p>
<p>Get started with your EDA now by exploring this voting records dataset numerically. It has been pre-loaded for you into a DataFrame called <code>df</code>.  Use pandas' <code>.head()</code>, <code>.info()</code>, and <code>.describe()</code> methods in the IPython Shell to explore the DataFrame, and select the statement below that is <strong>not</strong> true.</p>
</div>

<ul>
<li><div class="dc-input-radio__text">The DataFrame has a total of <code>435</code> rows and <code>17</code> columns.</div></li>
<li><div class="dc-input-radio__text">Except for <code>'party'</code>, all of the columns are of type <code>int64</code>.</div></li>
<li><div class="dc-input-radio__text">The first two rows of the DataFrame consist of votes made by Republicans and the next three rows consist of votes made by Democrats.</div></li>
<strong><li><div class="dc-input-radio__text">There are 17 <em>predictor variables</em>, or <em>features</em>, in this DataFrame.</div></li></strong>
<li><div class="dc-input-radio__text">The target variable in this DataFrame is <code>'party'</code>.</div></li>
</ul>

<p class="">Great work! The number of columns in the DataFrame is not equal to the number of features. One of the columns - <code>'party'</code> is the target variable.</p>

### Visual EDA


<div class>
<p>The Numerical EDA you did in the previous exercise gave you some very important information, such as the names and data types of the columns, and the dimensions of the DataFrame. Following this with some visual EDA will give you an even better understanding of the data. In the video, Hugo used the <code>scatter_matrix()</code> function on the Iris data for this purpose. However, you may have noticed in the previous exercise that all the features in this dataset are binary; that is, they are either 0 or 1. So a different type of plot would be more useful here, such as <a href="http://seaborn.pydata.org/generated/seaborn.countplot.html">Seaborn's <code>countplot</code></a>.</p>
<p>Given on the right is a <code>countplot</code> of the <code>'education'</code> bill, generated from the following code:</p>
<pre><code>plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
</code></pre>
<p>In <code>sns.countplot()</code>, we specify the x-axis data to be <code>'education'</code>, and hue to be <code>'party'</code>. Recall that <code>'party'</code> is also our target variable. So the resulting plot shows the difference in voting behavior between the two parties for the <code>'education'</code> bill, with each party colored differently. We manually specified the color to be <code>'RdBu'</code>, as the Republican party has been traditionally associated with red, and the Democratic party with blue.</p>
<p>It seems like Democrats voted resoundingly <em>against</em> this bill, compared to Republicans. This is the kind of information that our machine learning model will seek to learn when we try to predict party affiliation solely based on voting behavior. An expert in U.S politics may be able to predict this without machine learning, but probably not instantaneously - and certainly not if we are dealing with hundreds of samples! </p>
<p>In the IPython Shell, explore the voting behavior further by generating countplots for the <code>'satellite'</code> and <code>'missile'</code> bills, and answer the following question: Of these two bills, for which ones do Democrats vote resoundingly in <em>favor</em> of, compared to Republicans? Be sure to begin your plotting statements for each figure with <code>plt.figure()</code> so that a new figure will be set up. Otherwise, your plots will be overlaid onto the same figure.</p>
</div>

<ul>
<li><code>'satellite'</code></li>
<li><code>'missile'</code></li>
<strong><li><div class="dc-input-radio__text">Both <code>'satellite'</code> and <code>'missile'</code>.</div></li></strong>
<li><div class="dc-input-radio__text">Neither <code>'satellite'</code> nor <code>'missile'</code>.</div></li>
</ul>

<p class="">Correct! Democrats voted in favor of both <code>'satellite'</code> <em>and</em> <code>'missile'</code></p>

## The classification challenge



### k-Nearest Neighbors: Fit


<div class>
<p>Having explored the Congressional voting records dataset, it is time now to build your first classifier. In this exercise, you will fit a k-Nearest Neighbors classifier to the voting dataset, which has once again been pre-loaded for you into a DataFrame <code>df</code>. </p>
<p>In the video, Hugo discussed the importance of ensuring your data adheres to the format required by the scikit-learn API. The features need to be in an array where each column is a feature and each row a different observation or data point - in this case, a Congressman's voting record. The target needs to be a single column with the same number of observations as the feature data. We have done this for you in this exercise. Notice we named the feature array <code>X</code> and response variable <code>y</code>: This is in accordance with the common scikit-learn practice.</p>
<p>Your job is to create an instance of a k-NN classifier with 6 neighbors (by specifying the <code>n_neighbors</code> parameter) and then fit it to the data. The data has been pre-loaded into a DataFrame called <code>df</code>.</p>
</div>

<li>Import <code>KNeighborsClassifier</code> from <code>sklearn.neighbors</code>.</li>

```python
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

```
<li>Create arrays <code>X</code> and <code>y</code> for the features and the target variable. Here this has been done for you. Note the use of <code>.drop()</code> to drop the target variable <code>'party'</code> from the feature array <code>X</code> as well as the use of the <code>.values</code> attribute to ensure <code>X</code> and <code>y</code> are NumPy arrays. Without using <code>.values</code>, <code>X</code> and <code>y</code> are a DataFrame and Series respectively; the scikit-learn API will accept them in this form also as long as they are of the right shape.</li>

```python
import pandas as pd
df = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_1939/datasets/votes-ch1.csv')
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

```
<li>Instantiate a <code>KNeighborsClassifier</code> called <code>knn</code> with <code>6</code> neighbors by specifying the <code>n_neighbors</code> parameter.</li>

```python
# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

```
<li>Fit the classifier to the data using the <code>.fit()</code> method.</li>

```python
# Fit the classifier to the data
knn.fit(X, y)
```

```
#> KNeighborsClassifier(n_neighbors=6)
```

<p class="">Excellent! Now that your k-NN classifier with 6 neighbors has been fit to the data, it can be used to predict the labels of new data points.</p>

### k-Nearest Neighbors: Predict


<div class>
<p>Having fit a k-NN classifier, you can now use it to predict the label of a new data point. However, there is no unlabeled data available since all of it was used to fit the model! You can still use the <code>.predict()</code> method on the <code>X</code> that was used to fit the model, but it is not a good indicator of the model's ability to generalize to new, unseen data. </p>
<p>In the next video, Hugo will discuss a solution to this problem. For now, a random unlabeled data point has been generated and is available to you as <code>X_new</code>. You will use your classifier to predict the label for this new data point, as well as on the training data <code>X</code> that the model has already seen. Using <code>.predict()</code> on <code>X_new</code> will generate 1 prediction, while using it on <code>X</code> will generate 435 predictions: 1 for each sample.</p>
<p>The DataFrame has been pre-loaded as <code>df</code>. This time, you will create the feature array <code>X</code> and target variable array <code>y</code> yourself.</p>
</div>

<li>Create arrays for the features and the target variable from <code>df</code>. As a reminder, the target variable is <code>'party'</code>.</li>

```python
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

```
<li>Instantiate a <code>KNeighborsClassifier</code> with <code>6</code> neighbors.</li>

```python
# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

```
<li>Fit the classifier to the data.</li>

```python
# Fit the classifier to the data
knn.fit(X, y)
```

```
#> KNeighborsClassifier(n_neighbors=6)
```
<li>Predict the labels of the training data, <code>X</code>.</li>

```python
# Predict the labels for the training data X: y_pred
y_pred = knn.predict(X)
```
<li>Predict the label of the new data point <code>X_new</code>.</li>

```python
import numpy as np
X_new = pd.DataFrame((np.random.rand(1,16))) 

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction)) 
```

```
#> Prediction: ['democrat']
```

<p class="">Great work! Did your model predict <code>'democrat'</code> or <code>'republican'</code>? How sure can you be of its predictions? In other words, how can you measure its performance? This is what you will learn in the next video.</p>

## Measuring model performance



### The digits recognition dataset


<div class>
<p>Up until now, you have been performing binary classification, since the target variable had two possible outcomes. Hugo, however, got to perform
multi-class classification in the videos, where the target variable could take on three possible outcomes. Why does he get to have all the fun?!
In the following exercises, you'll be working with the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> digits recognition dataset, which has
10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn's included datasets, and that is the one we will use in this exercise. </p>
<p>Each sample in this scikit-learn dataset is an 8x8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black. Recall that scikit-learn's built-in datasets are of type <code>Bunch</code>, which are dictionary-like objects. Helpfully for the MNIST dataset, scikit-learn provides an <code>'images'</code> key in addition to the <code>'data'</code> and <code>'target'</code> keys that you have seen with the Iris data. Because it is a 2D array of the images corresponding to each sample, this <code>'images'</code> key is useful for visualizing the images, as you'll see in this exercise (for more on plotting 2D arrays, see <a href="https://www.datacamp.com/courses/introduction-to-data-visualization-with-python">Chapter 2</a> of DataCamp's course on Data Visualization with Python). On the other hand, the <code>'data'</code> key contains the feature array - that is, the images as a flattened array of 64 pixels.</p>
<p>Notice that you can access the keys of these <code>Bunch</code> objects in two different ways: By using the <code>.</code> notation, as in <code>digits.images</code>, or the <code>[]</code> notation, as in <code>digits['images']</code>. </p>
<p>For more on the MNIST data, check out <a href="https://campus.datacamp.com/courses/importing-data-in-python-part-1/introduction-and-flat-files-1?ex=10">this exercise</a> in Part 1 of DataCamp's Importing Data in Python course. There, the full version of the MNIST dataset is used, in which the images are 28x28. It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.</p>
</div>

<li>Import <code>datasets</code> from <code>sklearn</code> and <code>matplotlib.pyplot</code> as <code>plt</code>.</li>

```python
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

```
<li>Load the digits dataset using the <code>.load_digits()</code> method on <code>datasets</code>.</li>

```python
# Load the digits dataset: digits
digits = datasets.load_digits()

```
<li>Print the keys and <code>DESCR</code> of digits.</li>

```python
# Print the keys and DESCR of the dataset
print(digits.keys())
```

```
#> dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
```

```python
print(digits.DESCR)

```

```
#> .. _digits_dataset:
#> 
#> Optical recognition of handwritten digits dataset
#> --------------------------------------------------
#> 
#> **Data Set Characteristics:**
#> 
#>     :Number of Instances: 1797
#>     :Number of Attributes: 64
#>     :Attribute Information: 8x8 image of integer pixels in the range 0..16.
#>     :Missing Attribute Values: None
#>     :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
#>     :Date: July; 1998
#> 
#> This is a copy of the test set of the UCI ML hand-written digits datasets
#> https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
#> 
#> The data set contains images of hand-written digits: 10 classes where
#> each class refers to a digit.
#> 
#> Preprocessing programs made available by NIST were used to extract
#> normalized bitmaps of handwritten digits from a preprinted form. From a
#> total of 43 people, 30 contributed to the training set and different 13
#> to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
#> 4x4 and the number of on pixels are counted in each block. This generates
#> an input matrix of 8x8 where each element is an integer in the range
#> 0..16. This reduces dimensionality and gives invariance to small
#> distortions.
#> 
#> For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
#> T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
#> L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
#> 1994.
#> 
#> .. topic:: References
#> 
#>   - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
#>     Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
#>     Graduate Studies in Science and Engineering, Bogazici University.
#>   - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
#>   - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
#>     Linear dimensionalityreduction using relevance weighted LDA. School of
#>     Electrical and Electronic Engineering Nanyang Technological University.
#>     2005.
#>   - Claudio Gentile. A New Approximate Maximal Margin Classification
#>     Algorithm. NIPS. 2000.
```
<li>Print the shape of <code>images</code> and <code>data</code> keys using the <code>.</code> notation.</li>

```python
# Print the shape of the images and data keys
print(digits.images.shape)
```

```
#> (1797, 8, 8)
```

```python
print(digits.data.shape)

```

```
#> (1797, 64)
```
<li>Display the 1011th image using <code>plt.imshow()</code>. This has been done for you, so hit submit to see which handwritten digit this happens to be!</li>

```python
# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-14-1.png" width="90%" style="display: block; margin: auto;" />

<p class="">Good job! It looks like the image in question corresponds to the digit '5'. Now, can you build a classifier that can make this prediction not only for this image, but for all the other ones in the dataset? You'll do so in the next exercise!</p>

### Train/Test Split + Fit/Predict/Accuracy


<div class><p>Now that you have learned about the importance of splitting your data into training and test sets, it's time to practice doing this on the digits dataset! After creating arrays for the features and target variable, you will split them into training and test sets, fit a k-NN classifier to the training data, and then compute its accuracy using the <code>.score()</code> method.</p></div>

<li>Import <code>KNeighborsClassifier</code> from <code>sklearn.neighbors</code> and <code>train_test_split</code> from <code>sklearn.model_selection</code>.</li>

```python
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

```
<li>Create an array for the features using <code>digits.data</code> and an array for the target using <code>digits.target</code>.</li>

```python
# Create feature and target arrays
X = digits.data
y = digits.target

```
<li>Create stratified training and test sets using <code>0.2</code> for the size of the test set. Use a random state of <code>42</code>. Stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset.</li>

```python
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

```
<li>Create a k-NN classifier with <code>7</code> neighbors and fit it to the training data.</li>

```python
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

```
<li>Compute and print the accuracy of the classifier's predictions using the <code>.score()</code> method.</li>

```python
# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
```

```
#> KNeighborsClassifier(n_neighbors=7)
```

```python
print(knn.score(X_test, y_test))
```

```
#> 0.9833333333333333
```

<p class="">Excellent work! Incredibly, this out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.</p>

### Overfitting and underfitting


<div class>
<p>Remember the model complexity curve that Hugo showed in the video? You will now construct such a curve for the digits dataset! In this exercise, you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, you will develop your intuition for overfitting and underfitting.</p>
<p>The training and testing sets are available to you in the workspace as <code>X_train</code>, <code>X_test</code>, <code>y_train</code>, <code>y_test</code>. In addition, <code>KNeighborsClassifier</code> has been imported from <code>sklearn.neighbors</code>.</p>
</div>

<li>Inside the for loop:<ul>
<li>Setup a k-NN classifier with the number of neighbors equal to <code>k</code>.</li>

<li>Fit the classifier with <code>k</code> neighbors to the training data.</li>

<li>Compute accuracy scores the training set and test set separately using the <code>.score()</code> method and assign the results to the <code>train_accuracy</code> and <code>test_accuracy</code> arrays respectively.</li>

</ul>
</li>

```python
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
```

```
#> KNeighborsClassifier(n_neighbors=1)
#> KNeighborsClassifier(n_neighbors=2)
#> KNeighborsClassifier(n_neighbors=3)
#> KNeighborsClassifier(n_neighbors=4)
#> KNeighborsClassifier()
#> KNeighborsClassifier(n_neighbors=6)
#> KNeighborsClassifier(n_neighbors=7)
#> KNeighborsClassifier(n_neighbors=8)
```


```python
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-24-3.png" width="90%" style="display: block; margin: auto;" />

<p class="">Great work! It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data. Now that you've grasped the fundamentals of classification, you will learn about regression in the next chapter!</p>

# Regression

<p class="chapter__description">
    In the previous chapter, you used image and political datasets to predict binary and multiclass outcomes. But what if your problem requires a continuous outcome? Regression is best suited to solving such problems. You will learn about fundamental concepts in regression and apply them to predict the life expectancy in a given country using Gapminder data.
  </p>
  
## Introduction to regression



### Which of the following is a regression problem?

<div class=""><p>Andy introduced regression to you using the Boston housing dataset. But regression models can be used in a variety of contexts to solve a variety of different problems.</p>
<p>Given below are four example applications of machine learning. Your job is to pick the one that is <em>best</em> framed as a <strong>regression</strong> problem.</p></div>

<ul>
<li><div class="">An e-commerce company using labeled customer data to predict whether or not a customer will purchase a particular item.</div></li>
<li><div class="">A healthcare company using data about cancer tumors (such as their geometric measurements) to predict whether a new tumor is benign or malignant.</div></li>
<li><div class="">A restaurant using review data to ascribe positive or negative sentiment to a given review.</div></li>
<strong><li><div class="">A bike share company using time and weather data to predict the number of bikes being rented at any given hour.</div></li></strong>
</ul>

<p class="dc-completion-pane__message dc-u-maxw-100pc">Great work! The target variable here - the number of bike rentals at any given hour - is quantitative, so this is best framed as a regression problem.</p>

### Importing data for supervised learning


<div class>
<p>In this chapter, you will work with <a href="https://www.gapminder.org/data/">Gapminder</a> data that we have consolidated into one CSV file available in the workspace as <code>'gapminder.csv'</code>. Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country's GDP, fertility rate, and population. As in Chapter 1, the dataset has been preprocessed.</p>
<p>Since the target variable here is quantitative, this is a regression problem. To begin, you will fit a linear regression with just one feature: <code>'fertility'</code>, which is the average number of children a woman in a given country gives birth to. In later exercises, you will use all the features to build regression models.</p>
<p>Before that, however, you need to import the data and get it into the form needed by scikit-learn. This involves creating feature and target variable arrays. Furthermore, since you are going to use only one feature to begin with, you need to do some reshaping using NumPy's <code>.reshape()</code> method. Don't worry too much about this reshaping right now, but it is something you will have to do occasionally when working with scikit-learn so it is useful to practice.</p>
</div>

<li>Import <code>numpy</code> and <code>pandas</code> as their standard aliases.</li>

```python
fn = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_2433/datasets/gapminder-clean.csv'
from urllib.request import urlretrieve
urlretrieve(fn, 'gapminder.csv')

# Import numpy and pandas
```

```
#> ('gapminder.csv', <http.client.HTTPMessage object at 0x7f829b409400>)
```

```python
import numpy as np
import pandas as pd

```
<li>Read the file <code>'gapminder.csv'</code> into a DataFrame <code>df</code> using the <code>read_csv()</code> function.</li>

```python
# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

```
<li>Create array <code>X</code> for the <code>'fertility'</code> feature  and array <code>y</code> for the <code>'life'</code> target variable.</li>

```python
# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
```

```
#> Dimensions of y before reshaping: (139,)
```

```python
print("Dimensions of X before reshaping: {}".format(X.shape))
```

```
#> Dimensions of X before reshaping: (139,)
```
<li>Reshape the arrays by using the <code>.reshape()</code> method and passing in <code>-1</code> and <code>1</code>.</li>

```python
# Reshape X and y
y_reshaped = y.reshape(-1, 1)
X_reshaped = X.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: ", y_reshaped.shape)
```

```
#> Dimensions of y after reshaping:  (139, 1)
```

```python
print("Dimensions of X after reshaping: ", X_reshaped.shape)
```

```
#> Dimensions of X after reshaping:  (139, 1)
```

<p class="">Great work! Notice the differences in shape before and after applying the <code>.reshape()</code> method. Getting the feature and target variable arrays into the right format for scikit-learn is an important precursor to model building.</p>

### Exploring the Gapminder data


<div class>
<p>As always, it is important to explore your data before building models. On the right, we have constructed a heatmap showing the correlation between the different features of the Gapminder dataset, which has been pre-loaded into a DataFrame as <code>df</code> and is available for exploration in the IPython Shell. Cells that are in green show positive correlation, while cells that are in red show negative correlation. Take a moment to explore this: Which features are positively correlated with <code>life</code>, and which ones are negatively correlated? Does this match your intuition? </p>
<p>Then, in the IPython Shell, explore the DataFrame using pandas methods such as <code>.info()</code>, <code>.describe()</code>, <code>.head()</code>. </p>
<p>In case you are curious, the heatmap was generated using <a href="http://seaborn.pydata.org/generated/seaborn.heatmap.html">Seaborn's heatmap function</a> and the following line of code, where <code>df.corr()</code> computes the pairwise correlation between columns:</p>
<p><code>sns.heatmap(df.corr(), square=True, cmap='RdYlGn')</code></p>
<p>Once you have a feel for the data, consider the statements below and select the one that is <strong>not</strong> true. After this, Hugo will explain the mechanics of linear regression in the next video and you will be on your way building regression models!</p>
</div>

<ul>
<li><div class="dc-input-radio__text">The DataFrame has <code>139</code> samples (or rows) and <code>9</code> columns.</div></li>
<li><div class="dc-input-radio__text"><code>life</code> and <code>fertility</code> are negatively correlated.</div></li>
<strong><li><div class="dc-input-radio__text">The mean of <code>life</code> is <code>69.602878</code>.</div></li></strong>
<li><div class="dc-input-radio__text"><code>GDP</code> and <code>life</code> are positively correlated.</div></li>
</ul>




```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve

fn = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_2433/datasets/gapminder-clean.csv'
urlretrieve(fn, 'gapminder.csv')
```

```
#> ('gapminder.csv', <http.client.HTTPMessage object at 0x7ff9bcca5400>)
```

```python
df = pd.read_csv('gapminder.csv')
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.xticks(rotation=90)
```

```
#> (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]), [Text(0.5, 0, 'population'), Text(1.5, 0, 'fertility'), Text(2.5, 0, 'HIV'), Text(3.5, 0, 'CO2'), Text(4.5, 0, 'BMI_male'), Text(5.5, 0, 'GDP'), Text(6.5, 0, 'BMI_female'), Text(7.5, 0, 'life'), Text(8.5, 0, 'child_mortality')])
```

```python
plt.yticks(rotation=0)
```

```
#> (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]), [Text(0, 0.5, 'population'), Text(0, 1.5, 'fertility'), Text(0, 2.5, 'HIV'), Text(0, 3.5, 'CO2'), Text(0, 4.5, 'BMI_male'), Text(0, 5.5, 'GDP'), Text(0, 6.5, 'BMI_female'), Text(0, 7.5, 'life'), Text(0, 8.5, 'child_mortality')])
```

```python
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-29-5.png" width="90%" style="display: block; margin: auto;" />



## The basics of linear regression



### Fit &amp; predict for regression


<div class>
<p>Now, you will fit a linear regression and predict life expectancy using just one feature. You saw Andy do this earlier using the <code>'RM'</code> feature of the Boston housing dataset. In this exercise, you will use the <code>'fertility'</code> feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is <code>'life'</code>. The array for the target variable has been pre-loaded as <code>y</code> and the array for <code>'fertility'</code> has been pre-loaded as <code>X_fertility</code>.</p>
<p>A scatter plot with <code>'fertility'</code> on the x-axis and <code>'life'</code> on the y-axis has been generated. As you can see, there is a strongly negative correlation, so a linear regression should be able to capture this trend. Your job is to fit a linear regression and then predict the life expectancy, overlaying these predicted values on the plot to generate a regression line. You will also compute and print the \(R^2\) score using scikit-learn's <code>.score()</code> method.</p>
</div>


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_2433/datasets/gapminder-clean.csv')

y = df['life'].values
X = df.drop('life', axis=1)

# Reshape to 1-D
y = y.reshape(-1, 1)
X_fertility = X['fertility'].values.reshape(-1, 1) 
```

<li>Import <code>LinearRegression</code> from <code>sklearn.linear_model</code>.</li>

```python
# Import LinearRegression
from sklearn.linear_model import LinearRegression

```
<li>Create a <code>LinearRegression</code> regressor called <code>reg</code>.</li>

```python
# Create the regressor: reg
reg = LinearRegression()

```
<li>Set up the prediction space to range from the minimum to the maximum of <code>X_fertility</code>. This has been done for you.</li>

```python
# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

```
<li>Fit the regressor to the data (<code>X_fertility</code> and <code>y</code>) and compute its predictions using the <code>.predict()</code> method and the <code>prediction_space</code> array.</li>

```python
# Fit the model to the data
reg.fit(X_fertility, y)

```

```
#> LinearRegression()
```
<li>Compute and print the \(R^2\) score using the <code>.score()</code> method.</li>

```python
# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

```

```
#> 0.6192442167740035
```
<li>Overlay the plot with your linear regression line. This has been done for you, so hit submit to see the result!</li>

```python
# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-36-5.png" width="90%" style="display: block; margin: auto;" />

<p class="">Fantastic! Notice how the line captures the underlying trend in the data. And the performance is quite decent for this basic regression model with only one feature!</p>

### Train/test split for regression


<div class>
<p>As you learned in Chapter 1, train and test sets are vital to ensure that your supervised learning model is able to generalize well to new data. This was true for classification models, and is equally true for linear regression models. </p>
<p>In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over <strong>all</strong> features. In addition to computing the \(R^2\) score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models. The feature array <code>X</code> and target variable array <code>y</code> have been pre-loaded for you from the DataFrame <code>df</code>.</p>
</div>

<li>Import <code>LinearRegression</code> from <code>sklearn.linear_model</code>, <code>mean_squared_error</code> from <code>sklearn.metrics</code>, and <code>train_test_split</code> from <code>sklearn.model_selection</code>.</li>

```python
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

```
<li>Using <code>X</code> and <code>y</code>, create training and test sets such that 30% is used for testing and 70% for training. Use a random state of <code>42</code>.</li>

```python
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

```
<li>Create a linear regression regressor called <code>reg_all</code>, fit it to the training set, and evaluate it on the test set.</li>

```python
# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
```

```
#> LinearRegression()
```

```python
y_pred = reg_all.predict(X_test)

```
<li>Compute and print the \(R^2\) score using the <code>.score()</code> method on the test set.</li>

<li>Compute and print the RMSE. To do this, first compute the Mean Squared Error using the <code>mean_squared_error()</code> function with the arguments <code>y_test</code> and <code>y_pred</code>, and then take its square root using <code>np.sqrt()</code>.</li>

```python
# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
```

```
#> R^2: 0.8380468731429358
```

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
```

```
#> Root Mean Squared Error: 3.2476010800377244
```

<p class="">Excellent! Using all features has improved the model score. This makes sense, as the model has more information to learn from. However, there is one potential pitfall to this process. Can you spot it? You'll learn about this as well how to better validate your models in the next video!</p>

## Cross-validation



### 5-fold cross-validation


<div class>
<p>Cross-validation is a vital step in evaluating a model. It maximizes the amount of data that is used to train the model, as during the course of training, the model is not only trained, but also tested on all of the available data.</p>
<p>In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn's <code>cross_val_score()</code> function uses \(R^2\) as the metric of choice for regression. Since you are performing 5-fold cross-validation, the function will return 5 scores. Your job is to compute these 5 scores and then take their average.</p>
<p>The DataFrame has been loaded as <code>df</code> and split into the feature/target variable arrays <code>X</code> and <code>y</code>. The modules <code>pandas</code> and <code>numpy</code> have been imported as <code>pd</code> and <code>np</code>, respectively.</p>
</div>

<li>Import <code>LinearRegression</code> from <code>sklearn.linear_model</code> and <code>cross_val_score</code> from <code>sklearn.model_selection</code>.</li>

```python
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

```
<li>Create a linear regression regressor called <code>reg</code>.</li>

```python
# Create a linear regression object: reg
reg = LinearRegression()

```
<li>Use the <code>cross_val_score()</code> function to perform 5-fold cross-validation on <code>X</code> and <code>y</code>.</li>

```python
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

```
<li>Compute and print the average cross-validation score. You can use NumPy's <code>mean()</code> function to compute the average.</li>

```python
# Print the 5-fold cross-validation scores
print(cv_scores)

# Print the average 5-fold cross-validation score
```

```
#> [0.81720569 0.82917058 0.90214134 0.80633989 0.94495637]
```

```python
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
```

```
#> Average 5-Fold CV Score: 0.8599627722793233
```

<p class="">Great work! Now that you have cross-validated your model, you can more confidently evaluate its predictions.</p>

### K-Fold CV comparison


<div class>
<p>Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.</p>
<p>In the IPython Shell, you can use <code>%timeit</code> to see how long each 3-fold CV takes compared to 10-fold CV by executing the following <code>cv=3</code> and <code>cv=10</code>:</p>
<pre><code>%timeit cross_val_score(reg, X, y, cv = ____)
</code></pre>
<p><code>pandas</code> and <code>numpy</code> are available in the workspace as <code>pd</code> and <code>np</code>. The DataFrame has been loaded as <code>df</code> and the feature/target variable arrays <code>X</code> and <code>y</code> have been created.</p>
</div>

<li>Import <code>LinearRegression</code> from <code>sklearn.linear_model</code> and <code>cross_val_score</code> from <code>sklearn.model_selection</code>. </li>

```python
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

```
<li>Create a linear regression regressor called <code>reg</code>.</li>

```python
# Create a linear regression object: reg
reg = LinearRegression()

```
<li>Perform 3-fold CV and then 10-fold CV. Compare the resulting mean scores.</li>

```python
# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv = 3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
```

```
#> 0.8718712782622107
```

```python
cvscores_10 = cross_val_score(reg, X, y, cv = 10)
print(np.mean(cvscores_10))
```

```
#> 0.8436128620131151
```

<p class="">Excellent! Did you use <code>%timeit</code> in the IPython Shell to see how much longer it takes 10-fold cross-validation to run compared to 3-fold cross-validation?</p>

## Regularized regression



### Regularization I: Lasso


<div class>
<p>In the video, you saw how Lasso selected out the <code>'RM'</code> feature as being the most important for predicting Boston house prices, while shrinking the coefficients of certain other features to 0. Its ability to perform feature selection in this way becomes even more useful when you are dealing with data involving thousands of features. </p>
<p>In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.</p>
<p>The feature and target variable arrays have been pre-loaded as <code>X</code> and <code>y</code>.</p>
</div>

<li>Import <code>Lasso</code> from <code>sklearn.linear_model</code>.</li>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_2433/datasets/gapminder-clean.csv')

y = df['life'].values
X = df.drop('life', axis=1).values

df_columns = df.drop('life', axis=1).columns
# Import Lasso
from sklearn.linear_model import Lasso

```
<li>Instantiate a Lasso regressor with an alpha of <code>0.4</code> and specify <code>normalize=True</code>. </li>

```python
# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

```
<li>Fit the regressor to the data and compute the coefficients using the <code>coef_</code> attribute.</li>

```python
# Fit the regressor to the data
lasso.fit(X, y)
```

```
#> Lasso(alpha=0.4, normalize=True)
#> 
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.
#> If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:
#> 
#> from sklearn.pipeline import make_pipeline
#> 
#> model = make_pipeline(StandardScaler(with_mean=False), Lasso())
#> 
#> If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:
#> 
#> kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}
#> model.fit(X, y, **kwargs)
#> 
#> Set parameter alpha to: original_alpha * np.sqrt(n_samples). 
#>   warnings.warn(
```


```python
# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
```

```
#> [-0.         -0.         -0.          0.          0.          0.
#>  -0.         -0.07087587]
```
<li>Plot the coefficients on the y-axis and column names on the x-axis. This has been done for you, so hit submit to view the plot!</li>

```python
# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
```

```
#> ([<matplotlib.axis.XTick object at 0x7f829b431be0>, <matplotlib.axis.XTick object at 0x7f829b431bb0>, <matplotlib.axis.XTick object at 0x7f829b431250>, <matplotlib.axis.XTick object at 0x7f829b964430>, <matplotlib.axis.XTick object at 0x7f829b964a00>, <matplotlib.axis.XTick object at 0x7f829b96c070>, <matplotlib.axis.XTick object at 0x7f829b96c670>, <matplotlib.axis.XTick object at 0x7f829b96cdc0>], [Text(0, 0, 'population'), Text(1, 0, 'fertility'), Text(2, 0, 'HIV'), Text(3, 0, 'CO2'), Text(4, 0, 'BMI_male'), Text(5, 0, 'GDP'), Text(6, 0, 'BMI_female'), Text(7, 0, 'child_mortality')])
```

```python
plt.margins(0.02)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-53-7.png" width="90%" style="display: block; margin: auto;" />

<p class="">Great work! According to the lasso algorithm, it seems like <code>'child_mortality'</code> is the most important feature when predicting life expectancy.</p>

### Regularization II: Ridge


<div class>
<p>Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.</p>
<p>Recall that lasso performs regularization by adding to the loss function a penalty term of the <em>absolute</em> value of each coefficient multiplied by some alpha. This is also known as \(L1\) regularization because the regularization term is the \(L1\) norm of the coefficients. This is not the only way to regularize, however. </p>
<p>If instead you took the sum of the <em>squared</em> values of the coefficients multiplied by some alpha - like in Ridge regression - you would be computing the \(L2\) norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated \(R^2\) scores for each, using this function that we have defined for you, which plots the \(R^2\) score as well as standard error for each alpha:</p>
<pre><code>def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
</code></pre>
<p>Don't worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the \(R^2\) score varies with different alphas, and to understand the importance of selecting the right value for alpha. You'll learn how to tune alpha in the next chapter.</p>
</div>

<li>Instantiate a Ridge regressor and specify <code>normalize=True</code>.</li>

```python
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

```
<li>Inside the <code>for</code> loop:<ul>
<li>Specify the alpha value for the regressor to use.</li>

<li>Perform 10-fold cross-validation on the regressor with the specified alpha. The data is available in the arrays <code>X</code> and <code>y</code>.</li>

<li>Append the average and the standard deviation of the computed cross-validated scores. NumPy has been pre-imported for you as <code>np</code>.</li>

</ul>
</li>

```python
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))
```
<li>Use the <code>display_plot()</code> function to visualize the scores and standard deviations.</li>

```python
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    
# Display the plot
display_plot(ridge_scores, ridge_scores_std)
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-59-9.png" width="90%" style="display: block; margin: auto;" />

<p class="">Great work! Notice how the cross-validation scores change with different alphas. Which alpha should you pick? How can you fine-tune your model? You'll learn all about this in the next chapter!</p>

# Fine-tuning your model

<p class="chapter__description">
    Having trained your model, your next task is to evaluate its performance. In this chapter, you will learn about some of the other metrics available in scikit-learn that will allow you to assess your model's performance in a more nuanced manner. Next, learn to optimize your classification and regression models using hyperparameter tuning.
  </p>
  
## How good is your model?



### Metrics for classification


<div class>
<p>In Chapter 1, you evaluated the performance of your k-NN classifier based on its accuracy. However, as Andy discussed, accuracy is not always an informative metric. In this exercise, you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report. </p>
<p>You may have noticed in the video that the classification report consisted of three rows, and an additional <em>support</em> column. The <em>support</em> gives the number of samples of the true response that lie in that class - so in the video example, the support was the number of Republicans or Democrats in the test set on which the classification report was computed. The <em>precision</em>, <em>recall</em>, and <em>f1-score</em> columns, then, gave the respective metrics for that particular class.</p>
<p>Here, you'll work with the <a href="https://www.kaggle.com/uciml/pima-indians-diabetes-database">PIMA Indians</a> dataset obtained from the UCI Machine Learning Repository. The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies. Therefore, it is a binary classification problem. A target value of <code>0</code> indicates that the patient does <em>not</em> have diabetes, while a value of <code>1</code> indicates that the patient <em>does</em> have diabetes. As in Chapters 1 and 2, the dataset has been preprocessed to deal with missing values.</p>
<p>The dataset has been loaded into a DataFrame <code>df</code> and the feature and target variable arrays <code>X</code> and <code>y</code> have been created for you. In addition, <code>sklearn.model_selection.train_test_split</code> and <code>sklearn.neighbors.KNeighborsClassifier</code> have already been imported.</p>
<p>Your job is to train a k-NN classifier to the data and evaluate its performance by generating a confusion matrix and classification report.</p>
</div>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_1939/datasets/diabetes.csv')

df.insulin.replace(0, np.nan, inplace=True)
df.triceps.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)

df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.mean()))
y = df['diabetes']
X = df.drop('diabetes', axis=1)
```

<li>Import <code>classification_report</code> and <code>confusion_matrix</code> from <code>sklearn.metrics</code>.</li>

```python

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

```
<li>Create training and testing sets with 40% of the data used for testing. Use a random state of <code>42</code>.</li>

```python
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

```
<li>Instantiate a k-NN classifier with <code>6</code> neighbors, fit it to the training data, and predict the labels of the test set.</li>









































































































































































































































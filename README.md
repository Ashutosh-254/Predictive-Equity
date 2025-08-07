## Index

List of Figures II

- 1 Introduction List of Tables II
   - 1.1 Motivation behind analyzing the dataset
- 2 Results
   - 2.1 Predictability of Classification Problems
      - 2.1.1 Were both classification problems equally predictable?
   - 2.2 Performance of Models
      - 2.2.1 Were all the tested models equally effective?
   - 2.3 Impact of Dimension Reduction
         - sitive or Negative) 2.3.1 Did dimension reduction have a meaningful impact? (Po-
   - 2.4 Recommendations Based on Results
      - 2.4.1 Given the motivation, what would you recommend?
   - 2.5 Visual Representation of Results
      - 2.5.1 At least one figure supporting your conclusions
- 3 Methods
   - 3.1 Hyperparameter Tuning
      - 3.1.1 What values did you pick as hyperparameters?
      - 3.1.2 How did you decide on these values?
   - 3.2 Dimension Reduction
      - 3.2.1 Methods of Dimension Reduction Used
      - 3.2.2 Explanation of Chosen Methods
- 4 Conclusion


## Figures

Number Title
1 Visualization of Model Performance
2 Multiple Line Chart for PCA Outcomes
3 Heatmap for Accuracy

##### II


## Tables

Number Title
1 Model Results
2 Model Outputs after PCA
3 Accuracy Comparison
4 Optimal Hyperparameter Values

##### III


## 1 Introduction

The analysis of the Adult Income dataset, which includes personal income
and demographic data, is the main goal of the study. This main objective of
the analysis is to create classification models that can forecast important cha-
racteristics like a person’s educational level and whether or not their annual
income surpasses$50,000. Understanding the variables influencing income and
educational attainment requires an understanding of these classification pro-
blems. The aim of this study is to investigate the effects of dimension reduction
techniques like PCA, effects of splitting the dataset into multiple proportions
for training and testing on model performance and to use machine learning
models to obtain insights into income prediction.

### 1.1 Motivation behind analyzing the dataset

The significance of income and education classification in numerous real-world
applications serves as the driving force behind the analysis of this dataset. Such
models, for instance, might be used by banks or credit card firms to forecast
revenue levels, which can aid in establishing credit limits and lending terms.
By modifying the financial offers according to people’s anticipated income
stability, this could help with risk management as well. Understanding the
distribution of schooling among various demographic groups may also be useful
to policymakers and educational institutions, since it may have an impact on
how resources are allocated and policies are made.


## 2 Results

The results of the models used to solve our two classification problems predic-
ting a person’s income and educational attainment—will be discussed in this
section. We will compare the predictability of the two classification problems,
evaluate the models’ performance, and investigate the effects of dimension re-
duction strategies. We will also provide suggestions based on the findings and
provide illustrations to back up our conclusions.

### 2.1 Predictability of Classification Problems

We will first examine the predictability of each categorization task, assessing
its difficulties and achievements.

#### 2.1.1 Were both classification problems equally predictable?

We concentrated on two different classification problems in this project:

1. Income Prediction: If a person makes more than$50,000 annually
2. Education Level Prediction: Classifying a person’s highest education

The intricacy of the variables affecting income, such as marital status, work
hours, and occupation, made the Income Prediction problem more difficult.
Predictions are made harder by the fact that these characteristics are frequent-
ly associated and that their relationship to income might be complicated.

However, compared to income, education is a more structured variable with
fewer potential outliers, making the Education Level Prediction problem so-
mewhat easier to predict. The classification task is made easier by the fact
that educational achievement frequently exhibits a more predictable pattern,
particularly when compared to age and occupation.

Although both tests had strong predictive abilities, the Income Prediction
task was less predictable than the Education Level Prediction task. This was
probably because income-related characteristics varied more.


### 2.2 Performance of Models

In order to forecast two target variables—income prediction (if a person’s an-
nual income surpasses$50,000) and education level prediction (classifying a
person’s highest level of education), we tried a number of classification mo-
dels. Decision Trees, Random Forest, K-Nearest Neighbors (KNN), Artificial
Neural Networks (ANN), Support Vector Machines (SVM), and Naive Bayes
are among the models that were assessed. Accuracy and runtime metrics were
used to assess each model’s performance after it was trained on multiple ver-
sions of the entire dataset using optimized hyperparameters.

#### 2.2.1 Were all the tested models equally effective?

Both classification issues did not yield the same level of effectiveness from the
models. Even though all of the models performed quite well, there were diffe-
rences in how well they predicted the degree of education and wealth.

```
Table 1 Model Results
```
The training and testing accuracies for each model in both classification tasks
are compiled in the Model Performance Table as highlighted in Table 1. It
offers a straightforward comparison of the predictions made by each model for
income and education. The models’ efficacy for the two classification tasks was
not comparable. Although each model performed well, there were differences
in the accuracy of their predictions on income and educational attainment.

The most successful methods were Random Forest and Artificial Neural Net-
works (ANN), with Random Forest outperforming ANN on both tests. Both
throughout training and testing, these models demonstrated improved accu-
racy, particularly in the Income Prediction task. However, Naive Bayes per-
formed the least, particularly when it came to estimating income, when it was
less accurate than other models. Both KNN and SVM had mediocre results;
KNN was less successful than ensemble techniques like Random Forest.


The training and test accuracies for the education and income prediction mo-
dels across different algorithms are contrasted in the bar chart in Fig 1. A clear
visual depiction of each model’s performance across various accuracy criteria
is given by the chart.

```
Fig. 1 Visualization of Model Performance
```
Table 1 in combination with Fig. 1 show that while some models have flawless
training accuracy, their test accuracy varies. This suggests that although the
models may fit training data well, they may not generalize as well to unknown
data.

### 2.3 Impact of Dimension Reduction

To lower the dataset’s dimensionality and evaluate its effect on model per-
formance, we employed Principal Component Analysis (PCA) in this project.
In order to capture the most variance in the data, PCA reduces the original
features into a smaller set of uncorrelated components. Our goal was to lower
computational costs while keeping the most crucial information for classifica-
tion by minimizing the number of features.


2.3.1 Did dimension reduction have a meaningful impact? (Positive
or Negative)

PCA-based dimension reduction had a mixed effect. By reducing the amount
of features, it helped lower computational costs, but not all models showed a
consistent improvement in model accuracy which can be seen in Table 2.

```
Table 2 Model Outputs after PCA
```
After using PCA, the accuracy of some models Random Forest and ANN in
particular improved or remained largely constant. The reduction of noise and
complexity appeared to help these models concentrate on the most important
features, which improved their capacity for generalization. Other models, in-
cluding KNN and Naive Bayes, saw a minor decline in performance.

```
Fig. 2 Multiple Line Chart for PCA Outcomes
```

It seemed that crucial information was lost during the dimension reduction
procedure in these models, which are more susceptible to feature interactions.
Fig. 2 helps to understand the effect that PCA had on all models under con-
sideration.

In conclusion, Random Forest and ANN models benefited from dimension
reduction, while Naive Bayes and KNN models suffered. Though its efficacy
varied depending on the particular model employed, PCA generally showed
promise in decreasing complexity.

### 2.4 Recommendations Based on Results

The following suggestions can be made based on the outcomes of the classi-
fication models, PCA-assisted dimension reduction, and model performance
evaluation

#### 2.4.1 Given the motivation, what would you recommend?

Model Selection:In terms of predicting income and education level, Ran-
dom Forest and Artificial Neural Networks (ANN) consistently performed bet-
ter than the other models. Given their excellent accuracy and capacity to
handle complex interactions, they are the best choices for this classification
assignment. Despite being computationally efficient, Naive Bayes performed
poorly, especially when it came to predicting revenue. For more intricate, high-
dimensional datasets like this one, Naive Bayes should be avoided. Although
KNN can be helpful when interpretability is important, it can also be compu-
tationally costly, particularly when working with larger datasets.

Impact of Dimension Reduction (PCA):Random Forest and ANN bene-
fited from the use of PCA since it simplified their algorithms and allowed them
to concentrate on the most important features, which increased or maintai-
ned their accuracy. However, dimension reduction resulted in marginally lower
performance for KNN and Naive Bayes, indicating that the models might rely
on feature interactions that are lost during PCA. If maintaining these links
is essential to the model’s performance under such circumstances, it might be
preferable to neglect dimensionality reduction.


Model Tuning:For models like SVM and KNN, whose performance can vary
greatly depending on the selection of parameters like the kernel (for SVM) and
the number of neighbors (for KNN), it is crucial to concentrate on hyperpa-
rameter optimization. Optimizing these models’ hyperparameters is essential
to enhancing their functionality. Although Random Forest and ANN typically
perform well with default parameters, a well-balanced hyperparameter search
can guarantee that these models produce the best results.

Recommendations:As more sophisticated models like Gradient Boosting or
XGBoost have been proven to perform better than Random Forest and ANN
in certain applications, further study with these models may be helpful. Model
performance could be further enhanced by exploring feature engineering and
including more significant features, specifically for tasks like income prediction.

### 2.5 Visual Representation of Results

The findings drawn from the classification models for the two tasks, namely,
income prediction and educational level prediction are illustrated in this sec-
tion.

#### 2.5.1 At least one figure supporting your conclusions

A thorough comparison of each model’s test accuracy for the two classification
tasks, with and without PCA, is shown in Table 3. The performance of each
model is clearly shown in the table, which also illustrates how dimensionality
reduction affects the precision of forecasts for both income and education.

```
Table 3 Accuracy Comparison
```

```
Fig. 3 Heatmap for Accuracy
```
To visually compare the test accuracy for each model in the two classification
problems, a heatmap was created. The heatmap, as shown in Fig. 3 makes it
simple to visualize model performance by displaying the accuracy ratings for
each model.

The heatmap makes it clear that while Naive Bayes demonstrated lesser ac-
curacy, especially for Income Prediction, Random Forest and ANN models
consistently outperformed the others in both tasks.

Therefore, the accuracy distribution across the models and the effect of PCA
on test accuracy for both tasks are better understood because of the heatmap.


## 3 Methods

The outcomes of the models applied to the two classification problems will be
covered in this section. The hyperparameter tuning procedure and the effects
of dimension reduction methods, particularly Principal Component Analysis
(PCA), on the outcomes will be discussed.

### 3.1 Hyperparameter Tuning

A crucial step in maximizing the effectiveness of machine learning models is
hyperparameter tuning. The procedure for choosing the best hyperparameters
for the models utilized in the tasks of income prediction and educational level
prediction is covered in detail in this section. Achieving high accuracy and
efficiency requires careful selection of hyperparameters, which regulate the
behavior and complexity of the model.

We conducted a systematic search for the ideal hyperparameter values to make
sure our models operate at their peak efficiency.

#### 3.1.1 What values did you pick as hyperparameters?

The optimal hyperparameter values chosen for each model in the two classi-
fication problems (both with and without PCA) are shown in Table 4. The
models’ performance during the cross-validation phase was used to determine
these values.

```
Table 4 Optimal Hyperparameter Values
```

#### 3.1.2 How did you decide on these values?

The ability of the hyperparameter values to balance the accuracy of the model
and its performance led to their selection. We chose a maximum depth of 10
for the decision trees since it avoided overfitting while preserving good accura-
cy. For Problem 1 and Problem 2, Random Forest’s ideal N Estimators values
were 250 and 200, respectively, offering the highest accuracy at a reasonable
training duration. With 11 neighbors, KNN outperformed the others, success-
fully balancing variance and bias. With five hidden units, the ANN model
reached its maximum accuracy while collecting enough complexity to prevent
overfitting.

For SVM, the linear kernel worked best for Problem 2, while the sigmoid
kernel was chosen for Problem 1. With alpha set as 5 for Problem 1 and 10
for Problem 2, Naive Bayes provided the best results in terms of smoothing
impact. To achieve reliable findings, each value was selected after a variety of
possibilities were tested and their effects on the models’ performance in both
tasks were assessed using cross-validation.

### 3.2 Dimension Reduction

In order to increase the effectiveness and performance of our machine lear-
ning models, we applied dimension reduction techniques. Repetitive or linked
characteristics are frequently found in high-dimensional datasets, which may
result in overfitting and higher processing requirements. We employed Princi-
pal Component Analysis (PCA), a popular method that minimizes the amount
of features while keeping the most crucial information in the data to address
this issue. The objective was to evaluate the effects of decreasing the datasets’
dimensionality on the effectiveness and performance of the models, both with
and without PCA.

#### 3.2.1 Methods of Dimension Reduction Used

With the goal to minimize the amount of features in our datasets, we used
Principal Component Analysis (PCA) for dimension reduction. PCA is a me-
thod that preserves as much variance as feasible while reducing the initial set
of features into a smaller number of uncorrelated variables known as princi-
pal components. We choose PCA due to its ability to effectively compress the
feature space, which can enhance model performance, particularly in high-
dimensional datasets.


For the tasks of predicting income and educational level, we used the mentio-
ned technique. The feature set was reduced to half in each instance, so that
50% of the variance was retained, and the models’ performance was assessed
both with and without PCA.

#### 3.2.2 Explanation of Chosen Methods

Finding a collection of orthogonal vectors, or principle components, that best
account for the dataset’s variation is how PCA operates. We can effectively
decrease the dimensionality by projecting the original data onto these com-
ponents, which are arranged according to the amount of variance they capture.
To determine the directions of maximum variance, the method starts by cal-
culating the data’s covariance matrix. The covariance matrix’s eigenvectors,
or major components, are then calculated. The major components that cap-
ture the greatest variance in the data are represented by the eigenvectors with
the highest eigenvalues. To create a lower-dimensional representation of the
dataset, we project the original data onto fewer of these eigenvectors.

We kept enough primary components for this analysis to account for half of the
variation. While lowering the number of dimensions, this balance guarantees
that a sizable portion of the data’s structure is maintained. The objective was
to minimize multicollinearity, a prevalent problem in high-dimensional data-
sets, and to eliminate the less significant features, which are frequently highly
linked. By removing noise and redundant attributes, dimensionality reduction
reduces overfitting while simultaneously increasing computing efficiency.

As PCA reduces the correlated variables to a smaller collection of uncorrela-
ted principle components, it works especially well when features are associated.
Using this method, models can concentrate on the most instructive features,
which could enhance prediction accuracy while preserving computational via-
bility.


## 4 Conclusion

The effectiveness of machine learning models for classification tasks was ex-
amined in this project in relation to the effects of hyperparameter tweaking
and dimensionality reduction approaches. The findings showed that while di-
mensionality reduction provided a trade-off between computing efficiency and
prediction accuracy, careful hyperparameter adjustment can greatly enhan-
ce model performance. Depending on the task and model, PCA’s effect on
model correctness varied, even though it decreased the feature space and in-
creased computing efficiency. With everything taken into account, this analysis
emphasizes how crucial it is to optimize model parameters and the possible
advantages of dimension reduction in enhancing machine learning applicati-
ons’ efficiency and performance.

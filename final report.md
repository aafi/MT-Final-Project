#Machine Translation
##Final Project Report
Anwesha Das

Deepti Panuganti

Dhruvil Shah

Aryeh Stiefel

###What We Did
For the default system we split up the sentences based on their assigned scores. We took every sentence with a score of $1$, then of `2` and then of `3`. For each of those clusters we computed the average value. Then, for each test sentence we computed its value and compared it to each of the cluster's values. The value that the sentence was closest to determined the score for that sentence.

<br/>
Then, for the baseline system we performed k nearest neighbors. We chose KNN because of its similarity to clustering, our default system. For each test sentence, we found the k closest values. Then, we took the majority score of the k nearest sentences and assigned it to the test sentence. We tried varying values of k from one to twenty one. We chose to stop increasing k because we stopped finding any significant improvement.

<br/>
We then began implementing various extensions. We got the ideas for them from the *Findings of the 2014 Workshop on Statistical Machine Translation* paper, by Bojar et. al. The first extension we implemented was Linear Regression, using the Python library, `scikit`. Using this method, on top of KNN, we were able to beat the baseline. Linear regression generated weights for the features and then recomputed KNN using the weighted values for each feature.

<br/>
The next extension we implemented was Support Vector Machines (SVMs) with radial basis function (RBF) kernel. We again used the Python library, `scikit`. We chose SVMs because it appeared in a lot of papers and yielded good results. We used grid search and three-fold-cross-validation to find the optimal parameters, $c$ and $\gamma$. We used three-fold-cross-validation because we tried various k-folds and three yielded the highest accuracy.

<br/>
The third extension we implemented was Decision Trees, again using the Python library, `scikit`.  First fit a linear regression model to the data and used the coefficients returned as weights for each feature. Then, using this weighted values for each feature we trained a decision tree, picking the features that yielded the highest information gain. We used the decision tree to then score the test sentences.

### Results
Default: 
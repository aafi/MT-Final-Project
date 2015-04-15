There are seven Python programs here (`-h` for usage):

+ `./default` performs clustering on the training data
+ `./baseline` performs k-nearest-neighbor
+ `./extension-1` performs linear regression with k-nearest-neighbors
+ `./extension-2` performs decision trees with linear regression
+ `./extension-3` performs support vector machine regression
+ `./extension-4` performs random forest
+ `./grade` compares the predicted score with the actual score and returns the accuracy

Any of the first six commands are designed to work in a pipeline with the last. For instance, this is a valid invocation:

```
./baseline | ./grade
```

In addition to the above files, there are similar Python programs:

+ `./nb` performs Naive Bayes classification
+ `./decision_tree` performs decision trees

They can be run in the same way as the first six.

<br/>
The `data/` directory contains one directory:

+ `train`, which contains
	+ `es-en_score.train`, which contains the score assigned to each sentence (one score per line)
	+ `es-en_source.train`, which contains 1050 Spanish sentences
	+ `es-en_target.train`, which contains 1050 English translations
	+ `train_features`, which contains the values of each of the 17 features for each sentence

as well as the `feature-list`, which contains the list of features being used

The `test-data/` directory contains one directory:

+ `test`, which contains
	+ `es-en_score.test`, which contains the true scores for each sentence (do not include this file when uploading as an assignment)
	+ `es-en_source.test`, which contains 450 Spanish sentences
	+ `es-en_target.test`, which contains 450 English translations
	+ `test_features`, which contains the values of each of the 17 features for each test sentence
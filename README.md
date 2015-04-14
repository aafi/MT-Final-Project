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
The `data/` directory contains two directories:

+ `train`
+ `train_out_of_domain`

as well as the `feature-list`.

The `test-data/` directory contains one directory:

+ `test`
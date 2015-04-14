#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict, Counter
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,grid_search
from sklearn import cross_validation


optparser = optparse.OptionParser()
optparser.add_option("-c", "--score", dest="score_train", default="data/train/es-en_score.train", help="Score train set")
optparser.add_option("-s", "--source", dest="source_train", default="data/train/es-en_source.train", help="Scourse train set")
optparser.add_option("-t", "--target", dest="target_train", default="data/train/es-en_target.train", help="Target train set")
optparser.add_option("-f", "--features", dest="feature_train", default="data/train/train_features", help="Feature train set")


(opts, _) = optparser.parse_args()

score = [float(s.strip()) for s in open(opts.score_train).readlines()]
source = [s.strip() for s in open(opts.source_train).readlines()]
target = [s.strip().split() for s in open(opts.target_train).readlines()]
features = [s.strip().split() for s in open(opts.feature_train).readlines()]

def get_label(feats, clf):
    label = clf.predict(feats)
    return label[0]

X_train = features
X_test = [s.strip().split() for s in open("data/test/test_features")]
y_train = score

# Create classifier
test_scores = {}
rfc = RandomForestClassifier()

# Set optimal parameter values using grid search and cross validation
parameters = {'n_estimators':[1,30], 'criterion':('gini', 'entropy'),'min_samples_leaf':[1,10]}

#Best cv value = 5
clf = grid_search.GridSearchCV(rfc,parameters,cv=5)

#Train classifier
clf.fit(X_train, y_train)

# Predict score
test_scores = {}
for i, feats in enumerate(X_test):
    test_scores[i] = int(get_label(feats, clf))

for score in test_scores.values():
    sys.stdout.write("%s\n" % score)
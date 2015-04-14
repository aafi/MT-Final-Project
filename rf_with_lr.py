#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict, Counter
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,grid_search
from sklearn import cross_validation
from sklearn import linear_model



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

def feat_val(feature_list):
    weighted_features = []
    for w, f in enumerate(feature_list):
        weighted_val = float(f) * weights[w]
        weighted_features.extend([weighted_val])
    return weighted_features

def get_label(feats, clf):
    label = clf.predict(feats)
    return label[0]

X_train = features
X_test = [s.strip().split() for s in open("test-data/test_features")]
y_train = score

#Linear Regression to weight the features
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)

weights = {}
for i, w in enumerate(ols.coef_):
    weights[i] = w
#sys.stdout.write("computing new feature values")
# Build new feature list with weighted values
weighted_features = list()
for i, feat in enumerate(features):
    weighted_features.append(feat_val(feat))

# Create classifier
test_scores = {}
rfc = RandomForestClassifier()

# Set optimal parameter values using grid search and cross validation
parameters = {'n_estimators':[1,30], 'criterion':('gini', 'entropy'),'min_samples_leaf':[1,10]}

#Best cv value = 5
clf = grid_search.GridSearchCV(rfc,parameters,cv=5)

#Train classifier
X_train = weighted_features
clf.fit(X_train, y_train)

# Predict score
test_scores = {}
for i, feats in enumerate(X_test):
    test_scores[i] = int(get_label(feats, clf))

for score in test_scores.values():
    sys.stdout.write("%s\n" % score)
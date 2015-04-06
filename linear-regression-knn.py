#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict, Counter
import operator
from sklearn import linear_model

optparser = optparse.OptionParser()
optparser.add_option("-c", "--score", dest="score_train", default="data/train/es-en_score.train", help="Score train set")
optparser.add_option("-s", "--source", dest="source_train", default="data/train/es-en_source.train", help="Scourse train set")
optparser.add_option("-t", "--target", dest="target_train", default="data/train/es-en_target.train", help="Target train set")
optparser.add_option("-f", "--features", dest="feature_train", default="data/train/train_features", help="Feature train set")

# k = 9 for best result!
optparser.add_option("-k", "--k", dest="k", type="int", default=9, help="k nearest neighbors")
(opts, _) = optparser.parse_args()

score = [float(s.strip()) for s in open(opts.score_train).readlines()]
source = [s.strip() for s in open(opts.source_train).readlines()]
target = [s.strip().split() for s in open(opts.target_train).readlines()]
features = [s.strip().split() for s in open(opts.feature_train).readlines()]

# Set the weights
weights = {}
for i, f in enumerate(features):
    weights[i] = 1

def get_distance(test, train):
    return abs(test - train)

def get_nearest_neighbors(sum_feats, k):
    dist_list = []
    for feat_sum, label in train_dict:
        dist = get_distance(sum_feats, feat_sum)
        dist_list.append((dist, label))

    sorted_list = sorted(dist_list, key = operator.itemgetter(0))
    return [tup[1] for tup in sorted_list[:k]]

def get_label(sum_feats):
    nearest_labels = get_nearest_neighbors(sum_feats, opts.k)
    mode = max(set(nearest_labels), key = nearest_labels.count)
    return mode

def feat_sum(feature_list):
    total = 0
    for w, f in enumerate(feature_list):
        total += float(f) * weights[w]
    return total

f = []
for feat_list in features:
    feats = []
    for feat in feat_list:
        feats.append(float(feat))
    f.append(feats)
# print f
X_train = f
Y_train = score
ols = linear_model.LinearRegression()
ols.fit(X_train, Y_train)

for i, w in enumerate(ols.coef_):
    weights[i] = w

# Read all sentences along with labels and features
train_dict = list()
for i, feat in enumerate(features):
    train_dict.append((feat_sum(feat), score[i]))

test_features = [s.strip().split() for s in open("data/test/test_features")]

test_scores = {}
for i, feats in enumerate(test_features):
    sum_feats = feat_sum(feats)
    test_scores[i] = int(get_label(sum_feats))

for score in test_scores.values():
    sys.stdout.write("%s\n" % score)
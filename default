#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-c", "--score", dest="score_train", default="data/train/es-en_score.train", help="Score train set")
optparser.add_option("-s", "--source", dest="source_train", default="data/train/es-en_source.train", help="Scourse train set")
optparser.add_option("-t", "--target", dest="target_train", default="data/train/es-en_target.train", help="Target train set")
optparser.add_option("-f", "--features", dest="feature_train", default="data/train/train_features", help="Feature train set")
(opts, _) = optparser.parse_args()

score = [s.strip() for s in open(opts.score_train).readlines()]
source = [s.strip() for s in open(opts.source_train).readlines()]
target = [s.strip().split() for s in open(opts.target_train).readlines()]
features = [s.strip().split() for s in open(opts.feature_train).readlines()]

# Set the weights
weights = {}
for i, f in enumerate(features):
    weights[i] = 1

def get_avg_score(score_list):
    feature_score = {}
    for i, index in enumerate(score_list):
        # sent = ones[index]
        feature = features[index]
        total = 0
        for w, f in enumerate(feature):
            total += float(f) * weights[w]
        feature_score[i] = total
    return sum(feature_score.values()) / float(len(feature_score))

# Extract sentences with corresponding score of 1, 2 or 3
ones = defaultdict()
twos = defaultdict()
threes = defaultdict()
for i, sent in enumerate(source):
    if int(score[i]) == 1:
        ones[i] = sent
    elif int(score[i]) == 2:
        twos[i] = sent
    elif int(score[i]) == 3:
        threes[i] = sent

ones_avg_score = get_avg_score(ones)
twos_avg_score = get_avg_score(twos)
threes_avg_score = get_avg_score(threes)

test_features = [s.strip().split() for s in open("test-data/test_features")]
total_feature_scores = {}
for i, feats in enumerate(test_features):
    total = 0
    for w, feat in enumerate(feats):
        total += float(feat) * weights[w]
    total_feature_scores[i] = total

test_scores = {}
for i, score in enumerate(total_feature_scores):
    ones_diff = abs(score - ones_avg_score)
    twos_diff = abs(score - twos_avg_score)
    threes_diff = abs(score - threes_avg_score)
    dummy = [ones_diff, twos_diff, threes_diff]
    smallest = min(dummy)
    if smallest == ones_diff:
        test_scores[i] = "1"
    elif smallest == twos_diff:
        test_scores[i] = "2"
    elif smallest == threes_diff:
        test_scores[i] = "3"

for score in test_scores.values():
    sys.stdout.write("%s\n" % score)
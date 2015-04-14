#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import math
import re

optparser = optparse.OptionParser()
optparser.add_option("-c", "--score", dest="score_train", default="data/train/es-en_score.train", help="Score train set")
optparser.add_option("-s", "--source", dest="source_train", default="data/train/es-en_source.train", help="Source train set")
optparser.add_option("-t", "--target", dest="target_train", default="data/train/es-en_target.train", help="Target train set")
optparser.add_option("-f", "--features", dest="feature_train", default="data/train/train_features", help="Feature train set")
(opts, _) = optparser.parse_args()

score = [s.strip() for s in open(opts.score_train).readlines()]
source = [s.strip() for s in open(opts.source_train).readlines()]
target = [s.strip().split() for s in open(opts.target_train).readlines()]
features = [s.strip().split() for s in open(opts.feature_train).readlines()]


def load_tokens(lines):
    tokens = []
    if type(lines) is str:
        s = lines.strip()
        a = re.findall(r"[\w']+|[.,!?;]", s)
        if len(a) == 1 and not a:
            pass
        else:
            tokens.extend(a)
    else:
        for line in lines:
            s = line.strip()
            a = s.split()
            if len(a) == 1 and not a:
                pass
            else:
                tokens.extend(a)
    return tokens


def log_probs(lines, smoothing):
    probs = defaultdict(str)
    all_tokens = []
    word_count = defaultdict(str)
    tokens = load_tokens(lines)
    all_tokens.extend(tokens)
    for token in all_tokens:
        if token not in word_count:
            word_count[token] = 1
        else:
            word_count[token] += 1
    total_count = sum(word_count.values())
    for word in word_count:
        numerator = word_count[word] + smoothing
        denominator = total_count + (smoothing * (len(word_count) + 1))
        probs[word] = math.log(numerator / denominator)
    unk_prob = smoothing / (total_count + (smoothing * (len(word_count) + 1)))
    probs["<UNK>"] = math.log(unk_prob)
    return probs, word_count


class Score(object):

    def __init__(self, smoothing):
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

        length_ones = len(load_tokens(ones.values()))
        length_twos = len(load_tokens(twos.values()))
        length_three = len(load_tokens(threes.values()))
        self.one_probs = log_probs(ones.values(), smoothing)[0]
        self.two_probs = log_probs(twos.values(), smoothing)[0]
        self.three_probs = log_probs(threes.values(), smoothing)[0]
        self.prob_is_one = length_ones / float(length_ones + length_twos + length_three)
        self.prob_is_two = length_twos / float(length_ones + length_twos + length_three)  # TODO confirm
        self.prob_is_three = length_three / float(length_ones + length_twos + length_three)
        # print self.prob_is_one, self.prob_is_two, self.prob_is_three
        self.smoothing = smoothing

    def get_score(self, sentence):
        _, word_count = log_probs(sentence, self.smoothing)
        # Is one
        total_ones_prob = 0
        for word in word_count:
            prob = self.one_probs[word]
            if not prob:
                prob = self.one_probs["<UNK>"]
            total_ones_prob += (prob * word_count[word])
        prob_one = self.prob_is_one * math.exp(total_ones_prob)

        total_twos_prob = 0
        for word in word_count:
            prob = self.two_probs[word]
            if not prob:
                prob = self.two_probs["<UNK>"]
            total_twos_prob += (prob * word_count[word])
        prob_two = self.prob_is_two * math.exp(total_twos_prob)
        # print prob_two

        total_threes_prob = 0
        for word in word_count:
            prob = self.one_probs[word]
            if not prob:
                prob = self.three_probs["<UNK>"]
            total_threes_prob += (prob * word_count[word])
        prob_three = self.prob_is_three * math.exp(total_threes_prob)

        # print prob_one, prob_two, prob_three

        highest = min(prob_one, prob_two, prob_three)
        if highest == prob_one:
            return "1"
        elif highest == prob_two:
            return "2"
        else:
            return "3"

scorer = Score(1e-5)
test_scores = {}
f = open("data/test/es-en_target.test")
for i, line in enumerate(f.readlines()):
    test_scores[i] = scorer.get_score(line)

for score in test_scores.values():
    sys.stdout.write("%s\n" % score)
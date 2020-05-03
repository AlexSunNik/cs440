"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
import sys


def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    #Consistently gives w the tag that was seen most often in the training dataset
    #For unseen words, should guess the tag that's seen the most often in the training dataset
    predicts = []
    #raise Exception("You must implement me")
    #Training on the train dataset

    #Build a dictionary of dictionaries
    #Each sub-dict records the frequency distribution over word tag
    train_dict = {}
    #Keep track of the most frequent tag
    total_tag_freq = {}
    for sentence in train:
        for word_tag in sentence:
            cur_tag_dict = train_dict.setdefault(word_tag[0], {})
            cur_tag_freq = cur_tag_dict.setdefault(word_tag[1], 0)
            cur_tag_dict[word_tag[1]] = cur_tag_freq + 1
            train_dict[word_tag[0]] = cur_tag_dict

            tag_freq = total_tag_freq.setdefault(word_tag[1], 0)
            total_tag_freq[word_tag[1]] = tag_freq + 1

    #Finish the dict building
    most_freq_tag = max(total_tag_freq.items(), key=lambda x: x[1])[0]
    #Predict the test dataset
    for sentence in test:
        cur_sentence_pred = []
        for word in sentence:
            if word not in train_dict:
                cur_sentence_pred.append((word, most_freq_tag))
            else:
                pred_tag = max(train_dict[word].items(), key=lambda x: x[1])[0]
                cur_sentence_pred.append((word, pred_tag))
        predicts.append(cur_sentence_pred)
    return predicts


#A custom data structure for running the viterbi algorithm
class trellis_node:
    def __init__(self, tag, value, prev_ptr):
        self.value = value
        self.tag = tag
        self.prev_ptr = prev_ptr


class viterbi_trellis:
    def __init__(self, tag_size, sentence_size, prior, alpha):
        #+1 for the prior probability
        self.trellis = np.zeros(
            shape=(tag_size, sentence_size + 1), dtype=object)
        prior_node = np.array([trellis_node(key, val, None)
                               for key, val in prior.items()])
        self.trellis[:, 0] = prior_node.T
        self.tag_size = tag_size
        self.alpha = alpha  # Smoothing factor
        self.cur_level = 1
    #Propagate for one layer

    def propagete(self, word):
        for i in range(self.tag_size):
            max_p = (None, -100000)  # tag, prob #for backtracking
            cur_tag = tags[i]
            #Get P(U_t | R_t) through laplace smoothing
            mea_p = np.log((measurement_model[cur_tag].setdefault(word, 0) + self.alpha) /
                           (sum(measurement_model[cur_tag].values()) + self.alpha * len(measurement_model[cur_tag])))
            for j in range(self.tag_size):
                cur_prev_node = self.trellis[j, self.cur_level - 1]
                cur_prev_tag = cur_prev_node.tag
                #Get P(R_t | R_t-1)
                trans_p = np.log((transition_model[cur_prev_tag].setdefault(cur_tag, 0) + self.alpha) /
                                 (sum(transition_model[cur_prev_tag].values()) + self.alpha * len(transition_model[cur_prev_tag])))
                cur_p = cur_prev_node.value + trans_p + mea_p
                if cur_p > max_p[1]:
                    max_p = (j, cur_p)

            self.trellis[i, self.cur_level] = trellis_node(
                cur_tag, max_p[1], max_p[0])

    def predict(self, sentence):
        predicts = []
        #Traverse through the trellis
        for word in sentence:
            self.propagete(word)
            self.cur_level = self.cur_level + 1

        level = self.cur_level - 1
        cur_node = max(self.trellis[:, level], key=lambda x: x.value)
        for i in reversed(range(len(sentence))):
            predicts.insert(0, (sentence[i], cur_node.tag))
            level -= 1
            cur_node = self.trellis[cur_node.prev_ptr, level]
        return predicts



def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    #Build the measurement and transition model in training process
    #A dictionary of dictionaries
    #Each subdictinary stores the frequency distribution over the next possible tag
    #Dict       keys: currrent tag to transition from --- values: a dictionary storing
    # the frequency distribution over the next possible tag
    #Subdict    keys: possible next tag --- values: the probability
    #global transition_model
    transition_model = {}
    #A dictionary of dictionaries
    #Each subdictinary stores the frequency distribution over words belonging to a certain tag
    #Dict       keys: tag  --- values: a dictionary storing
    # the word distribution over the key tag
    #Subdict    keys: word --- values: the probability
    #global measurement_model
    measurement_model = {}

    #global tags
    tags = []
    #Prior probability for each tag
    prior = {}
    for sentence in train:
        for i in range(len(sentence)):
            #Build the transition model
            if i != len(sentence) - 1:
                cur_trans_dict = transition_model.setdefault(
                    sentence[i][1], {})
                next_tag_count = cur_trans_dict.setdefault(sentence[i+1][1], 0)
                cur_trans_dict[sentence[i+1][1]] = next_tag_count + 1
                transition_model[sentence[i][1]] = cur_trans_dict
            #Build the measurement model
            cur_mea_dict = measurement_model.setdefault(sentence[i][1], {})
            tag_word_count = cur_mea_dict.setdefault(sentence[i][0], 0)
            cur_mea_dict[sentence[i][0]] = tag_word_count + 1
            measurement_model[sentence[i][1]] = cur_mea_dict

            tag_count = prior.setdefault(sentence[i][1], 0)
            prior[sentence[i][1]] = tag_count + 1
    #Calculate the log prior probability
    total_word_count = sum(prior.values())

    for tag_key in prior.keys():
        prior[tag_key] = np.log(prior[tag_key] / total_word_count)
    tags = list(prior.keys())

    total_tag_word_count = {tag: sum(measurement_model[tag].values()) for tag in tags}
    trans_tag_sum = {tag: sum(transition_model[tag].values()) for tag in tags}
    cur_trellis = viterbi_trellis(len(prior), len(test[10]), prior, 1)
    #Predict by building the trellis
    predicts = []
    alpha = 1
    prior_dict = {key: [value, None] for key, value in prior.items()}

    for i in range(len(test)):
        if i % 100 == 0:
            print(i / len(test))
        sentence = test[i]
        trellis = {}
        trellis[0] = prior_dict
        for i in range(len(sentence)):
            trellis[i+1] ={}
            word = sentence[i]
            for cur_tag in tags:
                max_p = [-100000, None]  # tag, prob #for backtracking
                mea_p = np.log((measurement_model[cur_tag].get(word, 0) + alpha) / \
                        (sum(measurement_model[cur_tag].values()) + alpha * len(measurement_model[cur_tag])))
                for prev_tag in tags:
                    trans_p = np.log((transition_model[prev_tag].setdefault(cur_tag, 0) + alpha) / \
                        (sum(transition_model[prev_tag].values()) + alpha * len(transition_model[prev_tag])))
                    cur_p = mea_p + trans_p + trellis[i][prev_tag][0]
                    if cur_p > max_p[0]:
                        max_p = [cur_p, prev_tag]
                trellis[i+1][cur_tag] = max_p
        #print(trellis)
        cur_tag_node = max( trellis[len(sentence)].items() ,key = lambda x: x[1][0])
        cur_sentence_pred = []
        for i in reversed(range(len(sentence))):
            cur_sentence_pred.insert(0, (sentence[i], cur_tag_node[0]))
            cur_tag_node = (cur_tag_node[1][1], trellis[i][cur_tag_node[1][1]])
        predicts.append(cur_sentence_pred)
    print(predicts[10])
    #raise Exception("You must implement me")
    return predicts


def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    raise Exception("You must implement me")
    return predicts

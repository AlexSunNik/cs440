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
    #Initial probability
    starting_prob = {}
    total_word = set()

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

            total_word.add(sentence[i][0])
        cur_tag_starting_prob = starting_prob.setdefault(sentence[0][1], 0)
        starting_prob[sentence[0][1]] = cur_tag_starting_prob + 1

    total_num_word = len(total_word)
    tags = list(measurement_model.keys())
    total_tag_word_count = {tag: sum(measurement_model[tag].values()) for tag in tags}
    trans_tag_sum = {tag: sum(transition_model[tag].values()) for tag in tags}
    #Predict by building the trellis
    predicts = []
    alpha = 0.1

    for i in range(len(test)):
        if i % 100 == 0:
            print(i / len(test))
        sentence = test[i]
        trellis = {}
        for i in range(len(sentence)):
             word = sentence[i]
             if i == 0:
                init_dict = {}
                for tag in tags:
                    start_p = np.log((starting_prob.get(tag, 0) + alpha) / (len(train) + alpha * len(tags)))
                    init_mea_p = np.log((measurement_model[tag].setdefault(word, 0) + alpha) / 
                        (total_tag_word_count[tag] + alpha * (total_num_word + 1)))
                    init_dict[tag] = [start_p + init_mea_p, None]
                trellis[i] = init_dict
                continue
             trellis[i] = {}
             for cur_tag in tags:
                max_p = [-sys.maxsize, None]  # tag, prob #for backtracking
                mea_p = np.log((measurement_model[cur_tag].setdefault(word, 0) + alpha) /
                               (total_tag_word_count[cur_tag] + alpha * (total_num_word + 1)))
                for prev_tag in tags:
                    trans_p = np.log((transition_model[prev_tag].setdefault(cur_tag, 0) + alpha) /
                                     (trans_tag_sum[prev_tag] + alpha * len(tags)))
                    cur_p = mea_p + trans_p + trellis[i-1][prev_tag][0]
                    if cur_p > max_p[0]:
                        max_p = [cur_p, prev_tag]
                trellis[i][cur_tag] = max_p
        cur_tag_node = max(trellis[len(sentence) - 1].items(), key=lambda x: x[1][0])
        cur_sentence_pred = []
        for i in reversed(range(len(sentence))):
            cur_sentence_pred.insert(0, (sentence[i], cur_tag_node[0]))
            if i == 0:
                break
            cur_tag_node = (cur_tag_node[1][1], trellis[i-1][cur_tag_node[1][1]])
        predicts.append(cur_sentence_pred)
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
    hapex_dict = {}

    #global tags
    tags = []
    #Initial probability
    starting_prob = {}
    total_word = set()

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
            #Build the hapex dictionary
            if sentence[i][0] not in total_word:
                #Add the word at the first encounter
                hapex_dict[sentence[i][0]] = sentence[i][1]
            else:
                #Encounter multiple times Check for rep
                if sentence[i][0] in hapex_dict:
                    del hapex_dict[sentence[i][0]]
            #Calculate the total number of unique words
            total_word.add(sentence[i][0])
        cur_tag_starting_prob = starting_prob.setdefault(sentence[0][1], 0)
        starting_prob[sentence[0][1]] = cur_tag_starting_prob + 1

    #Build the hapex count dict
    hapex_count_dict = {}
    for tag in hapex_dict.values():
        cur_count = hapex_count_dict.setdefault(tag, 0)
        hapex_count_dict[tag] = cur_count + 1

    total_num_word = len(total_word)
    tags = list(measurement_model.keys())
    total_tag_word_count = {
        tag: sum(measurement_model[tag].values()) for tag in tags}
    trans_tag_sum = {tag: sum(transition_model[tag].values()) for tag in tags}
    #Predict by building the trellis
    predicts = []
    alpha = 0.01
    print(alpha)
    for i in range(len(test)):
        if i % 100 == 0:
            print(i / len(test))
        sentence = test[i]
        trellis = {}
        for i in range(len(sentence)):
            word = sentence[i]
            if i == 0:
                init_dict = {}
                for tag in tags:
                    start_p = np.log(
                        (starting_prob.get(tag, 0) + alpha) / (len(train) + alpha * len(tags)))
                    init_hapex_p = ((hapex_count_dict.setdefault(tag, 0) + alpha) /
                                          (len(hapex_dict) + alpha * len(tags)))
                    init_mea_p = np.log((measurement_model[tag].setdefault(word, 0) + alpha * init_hapex_p) /
                                        (total_tag_word_count[tag] + alpha * init_hapex_p * (total_num_word + 1)))
                    init_dict[tag] = [start_p + init_mea_p, None]
                trellis[i] = init_dict
                continue
            trellis[i] = {}
            for cur_tag in tags:
               max_p = [-sys.maxsize, None]  # tag, prob #for backtracking
               hapex_p = ((hapex_count_dict.setdefault(cur_tag, 0) + alpha) /
                                         (len(hapex_dict) + alpha * len(tags)))
               mea_p = np.log((measurement_model[cur_tag].setdefault(word, 0) + hapex_p * alpha) /
                              (total_tag_word_count[cur_tag] + hapex_p * alpha * (total_num_word + 1)))
               for prev_tag in tags:
                   trans_p = np.log((transition_model[prev_tag].setdefault(cur_tag, 0) + alpha) /
                                    (trans_tag_sum[prev_tag] + alpha * len(tags)))
                   cur_p = mea_p + trans_p + trellis[i-1][prev_tag][0]
                   if cur_p > max_p[0]:
                       max_p = [cur_p, prev_tag]
               trellis[i][cur_tag] = max_p
        cur_tag_node = max(
            trellis[len(sentence) - 1].items(), key=lambda x: x[1][0])
        cur_sentence_pred = []
        for i in reversed(range(len(sentence))):
            cur_sentence_pred.insert(0, (sentence[i], cur_tag_node[0]))
            if i == 0:
                break
            cur_tag_node = (cur_tag_node[1][1],
                            trellis[i-1][cur_tag_node[1][1]])
        predicts.append(cur_sentence_pred)
    #raise Exception("You must implement me")
    return predicts

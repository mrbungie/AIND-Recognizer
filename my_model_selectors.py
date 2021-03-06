import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        best_bic = float('Inf') # to start the selector
        best_model = None # just in case we don't find any
        # for each num_component to test
        for num_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # we train the model and get its log-likelihood
                current_model = self.base_model(num_components)
                logL = current_model.score(self.X, self.lengths)
                # number of parameters according to https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/15
                bic = -2 * logL + (num_components ** 2 + 2 * num_components * current_model.n_features - 1 ) * np.log(len(self.sequences))
                # shall it be better than the best_bic yet, it becomes the best_bic and we select this model as the best_model
                if bic < best_bic:
                    best_bic = bic
                    best_model = current_model
            except:
                # copied from the function above (base_model)
                if self.verbose:
                    print("failure on {} with {} states, continuing".format(self.this_word, num_components))
                pass
        # we return the best_model
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        best_dic = float('-Inf') # to start the selector
        best_model = None # just in case we find no model

        other_words = [value for key, value in self.hwords.items() if key != self.this_word] # 
        for num_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # we train for the current word and we get the logL
                current_model = self.base_model(num_components)
                Xi_logL = current_model.score(self.X, self.lengths)
                sum_other_Xi_logL = float(0)
                # we score every other class, calculating logL for competing words
                for word in other_words:
                    sum_other_Xi_logL +=  current_model.score(word[0], word[1])
                dic = Xi_logL - (1/len(other_words))*sum_other_Xi_logL
                # according to the paper, if the model presents a greater criterion value, it's a better model 
                # shall it be better than the best_dic yet, it becomes the best_dic and we select this model as the best_model
                if dic > best_dic:
                    best_dic = dic
                    best_model = current_model
            except:
                # copied from the function above (base_model)
                if self.verbose:
                    print("failure on {} with {} states, continuing".format(self.this_word, num_components))
                continue
        # we return the best_model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    
    def select(self):
        best_logLavg = float('-Inf') # to start the selector
        best_model = None # just in case we don't find a model
        best_num_components = None # just in case we don't find a model

        def cv_loop(num_components):
            """ CV loop helper function """
            logLs = []
            # I thought I needed to do something like this (as it was failing for FISH) but I confirmed it using the forums: https://discussions.udacity.com/t/selectorcv-fails-to-train-fish/338796
            split_method = KFold(n_splits=min(3,len(self.sequences))) 
            # for each fold
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    # we get X and lengths for both train and test set
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    # we train the model
                    current_model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    # and we append the logL to our list
                    logLs.append(current_model.score(X_test, lengths_test))
                except:
                    # copied from the function above (base_model)
                    if self.verbose:
                        print("failure on {} with {} states, continuing".format(self.this_word, num_components))
                    continue
            # if we found at least one logL we return the average
            if len(logLs) > 0:
                return (sum(logLs)/len(logLs))
            else:
                return float('-Inf')

        for num_components in range(self.min_n_components, self.max_n_components+1):
            if len(self.sequences) > 1:
                # in case CV is possible (>1 sequences) we do the cv loop
                logLavg = cv_loop(num_components)
            else:
                # if <1 sequences, we train using all the data (no cv possible)
                logLavg = float('-Inf')
                try:
                    current_model = self.base_model(num_components)
                    logLavg = current_model.score(self.X, self.lengths)
                except:
                    pass

            # we compare the current logLavg with the best one yet, if it's better, we assign it as the best logLavg and set
            # the number of components as the best yet
            if logLavg > best_logLavg:
                best_logLavg = logLavg
                best_num_components = num_components

        # if we found the best number of components, we create a new model
        if best_num_components is not None:
            best_model = self.base_model(best_num_components)

        return best_model
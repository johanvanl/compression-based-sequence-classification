from collections import Counter

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV

from compression_classifier.classifier.helpers.list import split_into_classes
from compression_classifier.classifier.conc_dict.kmer_scoring import EScoring, get_class_scores
from compression_classifier.classifier.conc_dict.dict_builder import ClassDictionaryBuilder
from compression_classifier.classifier.conc_dict.lz77 import HashBasedMatchFinder, generate_sequences

class LZ77ConcDictVectorizer(BaseEstimator, TransformerMixin):

    _parameter_constraints = {
        'e_scoring' : [EScoring],
        'dict_size' : [int],
        'k' : [int],
        'segment_size' : [int],
        'min_match' : [int]
    }

    def __init__(self,
                 e_scoring : EScoring = EScoring.TF,
                 dict_size : int = 2**16,
                 k : int = -1,
                 segment_size : int = -1,
                 min_match : int = 4,
                 class_scores : list[Counter] = None):
        self.e_scoring = e_scoring
        self.dict_size = dict_size
        self.k = k
        self.segment_size = segment_size
        self.min_match = min_match
        self.class_scores = class_scores

    def _more_tags(self):
        return { 'stateless' : False }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y): 
        check_classification_targets(y)
        assert self.k > 0 and self.segment_size > 0, \
            'k and segment_size needs to be set!'

        cls_samples = split_into_classes(X, y)
        self.num_classes_ = len(cls_samples)
        
        # No class scores passed, built them
        if self.class_scores is None:
            self.class_scores = []
            for cnt in get_class_scores(self.e_scoring, cls_samples, k=self.k):
                self.class_scores.append(cnt.build_positive_counter())

        self.matchers_ = []
        for class_idx in range(self.num_classes_):
            dict_builder = ClassDictionaryBuilder(cls_samples[class_idx],
                                                  self.class_scores[class_idx],
                                                  dict_size=self.dict_size, k=self.k,
                                                  segment_size=self.segment_size)
            self.matchers_.append(HashBasedMatchFinder(dict_builder.build(),
                                                       min_match_len=self.min_match,
                                                       max_match_len=self.segment_size))
        return self

    def transform(self, X):
        '''
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
        '''
        check_is_fitted(self)

        X_transformed = [[] for _ in range(len(X))]
        # Class first as we are producing long position based vectors
        for class_idx in range(self.num_classes_):
            for i in range(len(X)):
                X_transformed[i].extend(self._build_features(X[i], self.matchers_[class_idx]))

        return np.array(X_transformed)
    
    def _build_features(self, sample : bytes, matcher : HashBasedMatchFinder) -> list:
        sample_length = len(sample)

        lz77_commands = generate_sequences(sample, matcher)

        literal_lengths, dict_match_idxs, dict_match_lengths = [], [], []
        for literals, match_idx, match_len in lz77_commands:
            literal_lengths.append(len(literals))
            if match_len > 0:
                dict_match_idxs.append(match_idx)
                dict_match_lengths.append(match_len)
        literal_lengths = np.array(literal_lengths)
        dict_match_idxs = np.array(dict_match_idxs)
        dict_match_lengths = np.array(dict_match_lengths)

        if len(dict_match_lengths) == 0:
            return [1, 0, 0, 0, 0, 0]
        
        features = []

        # Compression Ratio
        comp_len = len(lz77_commands) * 3
        comp_len += np.sum(literal_lengths)
        comp_len += np.sum((np.clip(literal_lengths - 15, a_min=-1, a_max=None) // 256) + 1)
        comp_len += np.sum((np.clip(dict_match_lengths - 19, a_min=-1, a_max=None) // 256) + 1)
        features.append(np.clip(comp_len, a_min=None, a_max=sample_length) / sample_length)

        # Count of Dictionary Matches
        features.append(len(dict_match_lengths) / sample_length)

        # Dictionary Coverage
        features.append(np.sum(dict_match_lengths) / sample_length)

        # Log+1 Weighted Average Match Length
        features.append(np.average(dict_match_lengths, weights=(np.log(dict_match_lengths)+1)) / sample_length)

        # Max Match
        features.append(np.max(dict_match_lengths) / sample_length)

        # Median Dictionary Offset
        features.append(np.median(dict_match_idxs))

        return features

class LZ77ConcDictFeatureClassifier(BaseEstimator, ClassifierMixin):

    _parameter_constraints = {
        'e_scoring' : [EScoring],
        'dict_size' : [int],
        'k' : [int],
        'segment_size' : [int],
        'min_match' : [int],
        'classifier_split' : [float],
        'n_jobs' : [int],
        'verbose' : [bool]
    }

    def __init__(self,
                 e_scoring : EScoring = EScoring.TF,
                 dict_size : int = 2**16,
                 k : int = -1,
                 segment_size: int = -1,
                 min_match : int = 4,
                 class_scores : list[Counter] = None,
                 classifier_split : float = 0.4,
                 n_jobs : int = 1,
                 verbose : bool = True):
        self.e_scoring = e_scoring
        self.dict_size = dict_size
        self.k = k
        self.segment_size = segment_size
        self.min_match = min_match
        self.class_scores = class_scores
        self.classifier_split = classifier_split
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y): 
        check_classification_targets(y)

        # Build dictionaries with full training data
        self.vectorizer_ = LZ77ConcDictVectorizer(e_scoring=self.e_scoring, dict_size=self.dict_size,
                                                  k=self.k, segment_size=self.segment_size,
                                                  min_match=self.min_match, class_scores=self.class_scores)
        self.vectorizer_.fit(X, y)

        # Fit logistic regression with less training data, avoiding costly compression
        X, _, y, _ = train_test_split(X, y, train_size=self.classifier_split,
                                      shuffle=True, stratify=y)
        train_vec = self.vectorizer_.transform(X)

        param_grid = [{ 'log_reg__penalty': ['l1'],
                        'log_reg__C' : [0.001, 0.01, 0.1, 1],
                        'log_reg__max_iter' : [7500],
                        'log_reg__solver' : ['saga'] }]
        pipeline = Pipeline([('scaler', StandardScaler()), ('log_reg', LogisticRegression())])
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        gscv = GridSearchCV(pipeline, param_grid, cv=cv, refit=True,
                            n_jobs=self.n_jobs, pre_dispatch=self.n_jobs)
        gscv.fit(train_vec, y)
        if self.verbose:
            print(f'Best Params, C: {gscv.best_params_["log_reg__C"]}')
        self.estimator_ = gscv.best_estimator_

        return self
    
    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(self.vectorizer_.transform(X))
    
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(self.vectorizer_.transform(X))

    def predict(self, X):
        check_is_fitted(self)
        return np.argmax(self.decision_function(X), axis=1)

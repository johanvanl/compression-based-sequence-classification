from abc import ABC
from collections import Counter

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV

from compression_classifier.classifier.helpers.list import split_into_classes

from compression_classifier.classifier.conc_dict.kmer_scoring import EScoring, get_class_scores
from compression_classifier.classifier.conc_dict.dict_builder import ClassDictionaryBuilder

from compression_classifier.compressor.factory import Lz4Compressor
from compression_classifier.compressor.base import CompressorList

class BaseLZ4ConcDictEstimator(ABC, BaseEstimator):

    _parameter_constraints = {
        'e_scoring' : [EScoring],
        'dict_size' : [int],
        'k' : [int],
        'segment_size' : [int]
    }

    def __init__(self,
                 e_scoring : EScoring = EScoring.TF,
                 dict_size : int = 2**16,
                 k : int = -1,
                 segment_size : int = -1,
                 class_scores : list[Counter] = None):
        self.e_scoring = e_scoring
        self.dict_size = min(dict_size, 2**16)
        self.k = k
        self.segment_size = segment_size
        self.class_scores = class_scores

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

        # Build Dictionaries and Compressors
        self.compressors_ = CompressorList()
        for class_idx in range(self.num_classes_):
            dict_builder = ClassDictionaryBuilder(cls_samples[class_idx],
                                                  self.class_scores[class_idx],
                                                  dict_size=self.dict_size, k=self.k,
                                                  segment_size=self.segment_size)
            self.compressors_.add(Lz4Compressor(level=12, dict=dict_builder.build()))

        return self

class LZ4ConcDictFeatureVectorizer(BaseLZ4ConcDictEstimator, TransformerMixin):
        
    def __init__(self,
                 e_scoring: EScoring = EScoring.TF,
                 dict_size: int = 2**16,
                 k: int = -1,
                 segment_size: int = -1,
                 class_scores: list[Counter] = None):
        super().__init__(e_scoring, dict_size, k, segment_size, class_scores)

    def _more_tags(self):
        return { 'stateless' : False }

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
                X_transformed[i].extend(self._build_features(X[i], self.compressors_.get(class_idx)))

        return np.array(X_transformed)
    
    def _build_features(self, sample : bytes, compressor : Lz4Compressor) -> list:
        sample_length = len(sample)

        compressed_bytes = compressor.compress(sample)
        sequences = compressor.get_lz4_sequences(compressed_bytes)
        dict_matches = compressor.get_dict_matches(sequences)

        dict_match_idxs, dict_match_lengths = [], []
        for match_idx, match_len in dict_matches:
            if match_len > 0:
                dict_match_idxs.append(match_idx)
                dict_match_lengths.append(match_len)
        dict_match_idxs = np.array(dict_match_idxs)
        dict_match_lengths = np.array(dict_match_lengths)

        features = []

        # Compression Ratio
        features.append(np.clip(len(compressed_bytes), a_min=None, a_max=sample_length) / sample_length)

        if len(dict_match_lengths) == 0:
            features.extend([0, 0, 0, 0, 0])
            return features

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

class LZ4ConcDictLengthClassifier(BaseLZ4ConcDictEstimator, ClassifierMixin):

    _parameter_constraints = {
        'e_scoring' : [EScoring],
        'dict_size' : [int],
        'k' : [int],
        'segment_size' : [int]
    }

    def __init__(self,
                 e_scoring = EScoring.TF,
                 dict_size = 2 ** 16,
                 k = -1,
                 segment_size = -1,
                 class_scores = None):
        super().__init__(e_scoring, dict_size, k, segment_size, class_scores)
    
    def decision_function(self, X):
        check_is_fitted(self)
        scores = np.zeros((len(X), self.num_classes_))
        for i in range(len(X)):
            for class_idx in range(self.num_classes_):
                scores[i, class_idx] = self.compressors_.get(class_idx).compressed_len(X[i])
            # Prevent negatives if the bytes expanded
            l = len(X[i])
            scores[i] = np.clip(scores[i], a_min=None, a_max=l) / l
        # Expecting a higher score to be better
        return 1 - scores
    
    def predict_proba(self, X):
        check_is_fitted(self)
        y_score = np.exp(self.decision_function(X))
        return y_score / np.sum(y_score, axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self)
        return np.argmax(self.decision_function(X), axis=1)

class LZ4ConcDictFeatureClassifier(BaseEstimator, ClassifierMixin):

    _parameter_constraints = {
        'e_scoring' : [EScoring],
        'dict_size' : [int],
        'k' : [int],
        'segment_size' : [int],
        'classifier_split' : [float],
        'n_jobs' : [int],
        'verbose' : [bool]
    }

    def __init__(self,
                 e_scoring: EScoring = EScoring.TF,
                 dict_size: int = 2**16,
                 k: int = -1,
                 segment_size: int = -1,
                 class_scores: list[Counter] = None,
                 classifier_split : float = 0.4,
                 n_jobs : int = 1,
                 verbose : bool = True):
        self.e_scoring = e_scoring
        self.dict_size = min(dict_size, 2**16)
        self.k = k
        self.segment_size = segment_size
        self.class_scores = class_scores
        self.classifier_split = classifier_split
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y): 
        check_classification_targets(y)

        # Build dictionaries with full training data
        self.vectorizer_ = LZ4ConcDictFeatureVectorizer(e_scoring=self.e_scoring,
                                                        dict_size=self.dict_size,
                                                        k=self.k,
                                                        segment_size=self.segment_size,
                                                        class_scores=self.class_scores)
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
        gscv = GridSearchCV(pipeline, param_grid, cv=cv, refit=True, n_jobs=self.n_jobs)
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

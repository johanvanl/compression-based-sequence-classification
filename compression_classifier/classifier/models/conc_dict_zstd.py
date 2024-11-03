from abc import ABC
from collections import Counter

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from compression_classifier.classifier.helpers.list import split_into_classes

from compression_classifier.classifier.conc_dict.kmer_scoring import EScoring, get_class_scores
from compression_classifier.classifier.conc_dict.dict_builder import ClassDictionaryBuilder

from compression_classifier.compressor.zstd import ZstdCompressor
from compression_classifier.compressor.base import CompressorList

class BaseZstdConcDictEstimator(ABC, BaseEstimator):

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
        self.dict_size = dict_size
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
            self.compressors_.add(ZstdCompressor(level=22, dict=dict_builder.build()))

        return self

class ZstdConcDictClassifier(BaseZstdConcDictEstimator, ClassifierMixin):

    def __init__(self,
                 e_scoring: EScoring = EScoring.TF,
                 dict_size: int = 2**16,
                 k: int = -1,
                 segment_size: int = -1,
                 class_scores: list[Counter] = None):
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

import numpy as np

import xxhash

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from compression_classifier.classifier.helpers.storage import NoCopyDict
from compression_classifier.classifier.helpers.system_id_bidict import DynamicNoCopySystemIdBiDict

class LzjdClassifier(ClassifierMixin, BaseEstimator):
    '''
    The LZJD Classifier/Estimator.
    (https://www.researchgate.net/publication/318916484_An_Alternative_to_NCD_for_Large_Sequences_Lempel-Ziv_Jaccard_Distance)
    '''

    _parameter_constraints = { 'n_jobs' : [int] }

    def __init__(self, n_jobs : int = 1):
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        '''
        Fit the AMDL classifier from the training dataset.
        '''
        check_classification_targets(y)

        self.num_classes_ = len(np.unique(y))

        # For mapping to numeric vectors
        self.bidict_ = DynamicNoCopySystemIdBiDict()

        self.sets_ = NoCopyDict()

        self.y_ = y
        self.X_ = []
        for x in X:
            id = self.bidict_.encode(x)
            self.sets_[id] = self._lz_set(x)
            self.X_.append(id)

        self.X_ = np.array(self.X_, dtype=np.int32).reshape(-1, 1)

        self.knn_ = KNeighborsClassifier(n_neighbors=1, algorithm='brute',
                                         n_jobs=self.n_jobs,
                                         metric=self._dist_metric)
        self.knn_.fit(self.X_, self.y_)

        return self
    
    def _lz_set(self, b : bytes) -> set:
        s = set()
        start = 0
        end = 1
        while end < len(b):
            bs = b[start:end]
            if bs not in s:
                s.add(bs)
                start = end
            end += 1

        int_set = set()
        for x in s:
            int_set.add(xxhash.xxh32_intdigest(x))

        return int_set
    
    def _dist_metric(self, x1 : np.ndarray, x2 : np.ndarray):
        s1 : set = self.sets_[int(x1[0])]
        s2 : set = self.sets_[int(x2[0])]
        return 1 - ( len(s1.intersection(s2)) / len(s1.union(s2)) )

    def predict(self, X):
        '''
        Predict the class labels for the provided data.
        '''
        check_is_fitted(self)

        encoding = []
        for x in X:
            id = self.bidict_.encode(x)
            self.sets_[id] = self._lz_set(x)
            encoding.append(id)
        encoding = np.array(encoding, dtype=np.uint32).reshape(-1, 1)
    
        return self.knn_.predict(encoding)

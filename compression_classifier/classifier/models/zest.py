import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from compression_classifier.classifier.helpers.list import split_into_classes
from compression_classifier.compressor.zstd import ZstdCompressor, train_zstd_dict

class ZestClassifier(ClassifierMixin, BaseEstimator):
    '''
    The Zest Classifier/Estimator.
    Written as a sklearn estimator from:
    https://github.com/facebookresearch/zest/tree/main
    '''

    _parameter_constraints = {
        'max_num_dicts' : [int],
        'verbose' : [bool]
    }

    def __init__(self,
                 max_num_dicts : int = 4,
                 verbose : bool = True):
        self.max_num_dicts = max_num_dicts
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        check_classification_targets(y)

        cls_samples = split_into_classes(X, y)
        self.num_classes_ = len(cls_samples)

        self._set_dict_sizes(cls_samples)
        self._build_dicts(cls_samples)

        return self

    def _set_dict_sizes(self, cls_samples) -> None:
        compressor = ZstdCompressor(level=22)
        min_size = 2**15
        max_size = 0
        for x in cls_samples:
            sz = compressor.compressed_len(b''.join(x))
            min_size = min(min_size, sz)
            max_size = max(max_size, sz)
        min_size = max(min_size // 2, 256)
        max_size = max(min_size, max_size)
        num_dicts = min(self.max_num_dicts, 1 + (max_size - min_size) // min_size)

        # From their source: "when dealing with corpora of very different sizes,
        # a multiplicative formula may work better"
        self.sizes_ = [(min_size + i * (max_size - min_size) // num_dicts) for i in range(num_dicts)]
        if self.verbose:
            print(f'Dictionary Sizes: {self.sizes_}')

    def _build_dicts(self, cls_samples):
        self.compressors_ = [[] for _ in range(self.num_classes_)]
        for class_idx in range(self.num_classes_):
            dicts = []
            for sz in self.sizes_:
                dicts.append(train_zstd_dict(samples=cls_samples[class_idx], dict_size=sz, level=22))
                for dict in dicts:
                    self.compressors_[class_idx].append(ZstdCompressor(level=22, dict=dict))
    
    def decision_function(self, X):
        check_is_fitted(self)
        scores = np.zeros((len(X), self.num_classes_))
        for class_idx in range(self.num_classes_):
            for i in range(len(X)):
                for compressor in self.compressors_[class_idx]:
                    scores[i, class_idx] += compressor.compressed_len(X[i])
        # Expecting a higher score to be better
        return -1 * scores

    def predict_proba(self, X):
        check_is_fitted(self)
        y_score = self.decision_function(X)
        y_score = y_score + np.abs(np.min(y_score)) + 1
        return y_score / np.sum(y_score, axis=1, keepdims=True)
    
    def predict(self, X):
        '''
        Predict the class labels for the provided data.
        '''
        check_is_fitted(self)
        return np.argmax(self.decision_function(X), axis=1)

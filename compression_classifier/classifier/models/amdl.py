from abc import ABC

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from compression_classifier.classifier.helpers.list import split_into_classes
from compression_classifier.classifier.helpers.storage import NoCopyDict
from compression_classifier.classifier.helpers.system_id_bidict import DynamicNoCopySystemIdBiDict

from compression_classifier.compressor.factory import ECompressor, max_lvl_compressor_factory, max_lvl_dict_compressor_factory

class BaseAmdlClassifier(ABC, BaseEstimator, ClassifierMixin):

    _parameter_constraints = {
        'e_compressor' : [ECompressor],
        'max_bytes' : [int],
        'n_jobs' : [int],
        'verbose' : [int]
    }

    def __init__(self,
                 e_compressor : ECompressor = ECompressor.LZ4,
                 max_bytes : int = -1,
                 n_jobs : int = 1,
                 verbose : bool = True):
        self.e_compressor = e_compressor
        self.max_bytes = max_bytes
        self.n_jobs = n_jobs
        self.verbose = verbose

        if self.max_bytes < 0:
            match e_compressor:
                case ECompressor.ZLIB:
                    self.max_bytes = 2**15
                case ECompressor.LZ4:
                    self.max_bytes = 2**16
                case ECompressor.ZSTD:
                    self.max_bytes = 2**18
            if self.verbose:
                print(f'Set Max Bytes to {self.max_bytes}')

    def _build_chunks(self, X : list[bytes], y : np.ndarray) -> tuple[list[bytes], np.ndarray]:
        cls_samples = split_into_classes(X, y)

        # If any individual sample is greater than the max_bytes
        # break it into smaller pieces
        for class_idx in range(self.num_classes_):
            i = 0
            while i < len(cls_samples[class_idx]):
                if len(cls_samples[class_idx][i]) > self.max_bytes:
                    cls_samples[class_idx].append(cls_samples[class_idx][i][self.max_bytes:])
                    cls_samples[class_idx][i] = cls_samples[class_idx][i][:self.max_bytes]
                i += 1

        # Create new X, y
        chunks = []
        chunk_labels = []
        for class_idx in range(self.num_classes_):
            chunk = []
            chunk_size = 0
            for C in cls_samples[class_idx]:
                if chunk_size + len(C) > self.max_bytes:
                    chunks.append(b''.join(chunk))
                    chunk_labels.append(class_idx)
                    chunk = []
                    chunk_size = 0
                chunk.append(C)
                chunk_size += len(C)
            if len(chunk) > 0:
                chunks.append(b''.join(chunk))
                chunk_labels.append(class_idx)

        return chunks, np.array(chunk_labels, dtype=np.uint32)
    
    def predict(self, X):
        check_is_fitted(self)

        encoding = []
        for x in X:
            encoding.append(self.bidict_.encode(x))
        encoding = np.array(encoding, dtype=np.uint32).reshape(-1, 1)
    
        return self.knn_.predict(encoding)

class AmdlClassifier(BaseAmdlClassifier):

    def __init__(self,
                 e_compressor: ECompressor = ECompressor.LZ4,
                 max_bytes: int = -1,
                 n_jobs: int = 1,
                 verbose : bool = True):
        super().__init__(e_compressor, max_bytes, n_jobs, verbose)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        check_classification_targets(y)

        self.num_classes_ = len(np.unique(y))

        # Transform training data
        chunks, self.y_ = self._build_chunks(X, y)

        # Set up single compressor and storage for training chunks
        self.compressor_ = max_lvl_compressor_factory(self.e_compressor)
        self.compressor_storage_ = NoCopyDict()

        # For mapping to numeric vectors
        self.bidict_ = DynamicNoCopySystemIdBiDict()

        # Store base compressed length
        self.X_ = []
        for x in chunks:
            id = self.bidict_.encode(x)
            self.X_.append(id)
            self.compressor_storage_[id] = self.compressor_.compressed_len(x)
        self.X_ = np.array(self.X_, dtype=np.int32).reshape(-1, 1)

        # Setup KNN
        self.knn_ = KNeighborsClassifier(n_neighbors=1,
                                         algorithm='brute',
                                         n_jobs=self.n_jobs,
                                         metric=self._dist_metric)
        self.knn_.fit(self.X_, self.y_)

        return self
    
    def _dist_metric(self, x1 : np.ndarray, x2 : np.ndarray):
        x1, x2 = int(x1[0]), int(x2[0])
        if x2 in self.compressor_storage_:
            c = self.bidict_.decode(x2) + self.bidict_.decode(x1)
            return self.compressor_.compressed_len(c) - self.compressor_storage_[x2]
        if x1 in self.compressor_storage_:
            c = self.bidict_.decode(x1) + self.bidict_.decode(x2)
            return self.compressor_.compressed_len(c) - self.compressor_storage_[x1]
        raise ValueError('One of the values needs to be in the Compressor Dict!')

class AmdlDictClassifier(BaseAmdlClassifier):

    def __init__(self,
                 e_compressor: ECompressor = ECompressor.LZ4,
                 max_bytes: int = -1,
                 n_jobs: int = 1,
                 verbose : bool = True):
        super().__init__(e_compressor, max_bytes, n_jobs, verbose)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        check_classification_targets(y)

        self.num_classes_ = len(np.unique(y))

        # Transform training data
        chunks, self.y_ = self._build_chunks(X, y)

        # For mapping to numeric vectors
        self.bidict_ = DynamicNoCopySystemIdBiDict()

        # Set up compressors
        self.X_ = []
        self.compressors_ = {}
        for x in chunks:
            id = self.bidict_.encode(x)
            self.X_.append(id)
            self.compressors_[id] = max_lvl_dict_compressor_factory(self.e_compressor, dict=x)
        self.X_ = np.array(self.X_, dtype=np.int32).reshape(-1, 1)

        # Setup KNN
        self.knn_ = KNeighborsClassifier(n_neighbors=1, algorithm='brute',
                                         n_jobs=self.n_jobs,
                                         metric=self._dist_metric)
        self.knn_.fit(self.X_, self.y_)

        return self
    
    def _dist_metric(self, x1 : np.ndarray, x2 : np.ndarray):
        x1, x2 = int(x1[0]), int(x2[0])
        if x2 in self.compressors_:
            return self.compressors_[x2].compressed_len(self.bidict_.decode(x1))
        if x1 in self.compressor_storage_:
            return self.compressors_[x1].compressed_len(self.bidict_.decode(x2))
        raise ValueError('One of the values needs to be in the Compressor Dict!')

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from compression_classifier.classifier.helpers.storage import NoCopyDict
from compression_classifier.classifier.helpers.system_id_bidict import DynamicNoCopySystemIdBiDict

from compression_classifier.compressor.factory import ECompressor, max_lvl_compressor_factory

class NcdClassifier(ClassifierMixin, BaseEstimator):
    '''
    The NCD Classifier/Estimator.
    '''

    _parameter_constraints = {
        'e_compressor' : [ECompressor],
        'n_neighbors' : [int],
        'weights' : [str],
        'algorithm' : [str],
        'n_jobs' : [int],
        'verbose' : [bool]
    }

    def __init__(self,
                 e_compressor : ECompressor = ECompressor.LZ4,
                 n_neighbors : int = 1,
                 weights : str = 'distance',
                 algorithm : str = 'brute',
                 n_jobs : int = 1,
                 verbose : bool = True):
        self.e_compressor = e_compressor
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        '''
        Fit the NCD classifier from the training dataset.
        '''
        check_classification_targets(y)

        self.num_classes_ = len(np.unique(y))

        self.compressor_ = max_lvl_compressor_factory(self.e_compressor)
        self.sample_storage_ = NoCopyDict()

        # For mapping to numeric vectors
        self.bidict_ = DynamicNoCopySystemIdBiDict()

        self.y_ = y
        self.X_ = []
        for x in X:
            id = self.bidict_.encode(x)
            self.X_.append(id)
            self.sample_storage_[id] = self.compressor_.compressed_len(x)

        self.X_ = np.array(self.X_, dtype=np.int32).reshape(-1, 1)
   
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                         algorithm=self.algorithm,
                                         n_jobs=self.n_jobs,
                                         metric=self._dist_metric)
        self.knn_.fit(self.X_, self.y_)

        return self
    
    def _dist_metric(self, x1 : np.ndarray, x2 : np.ndarray) -> float:
        x1, x2 = int(x1[0]), int(x2[0])
    
        c1 = self.sample_storage_[x1]
        c2 = self.sample_storage_[x2]

        cc = self.compressor_.compressed_len(self.bidict_.decode(x2) + self.bidict_.decode(x1))
            
        assert c1 > 0 and c2 > 0 and cc > 0, \
                'Compressed lengths need to be positive!'
        
        return ( cc - min(c1, c2) ) / max(c1, c2)

    def predict(self, X):
        check_is_fitted(self)

        encoding = []
        for x in X:
            id = self.bidict_.encode(x)
            self.sample_storage_[id] = self.compressor_.compressed_len(x)
            encoding.append(id)
        encoding = np.array(encoding, dtype=np.uint32).reshape(-1, 1)

        return self.knn_.predict(encoding)

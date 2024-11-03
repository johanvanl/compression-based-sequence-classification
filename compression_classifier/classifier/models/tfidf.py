from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

def choose_k_vals(unique_tokens : int) -> list[int]:
    if unique_tokens < 20:
        return [8, 10]
    elif unique_tokens < 128:
        return [6, 8]
    elif unique_tokens < 200:
        return [5]
    return [4]

class BaseTfIdfClassifier(ABC, BaseEstimator, ClassifierMixin):
    '''
    Base Tf-Idf Classifier/Estimator.
    '''

    _parameter_constraints = {
        'k' : [int],
        'min_df' : [int],
        'use_idf' : [bool],
        'smooth_idf' : [bool],
        'sublinear_tf' : [bool]
    }

    def __init__(self,
                 k : int = 6,
                 min_df : int = 2,
                 use_idf : bool = True,
                 smooth_idf : bool = True,
                 sublinear_tf : bool = False):
        self.k = k
        self.min_df = min_df
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    @abstractmethod
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        pass

    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)

class TfIdfSvmClassifier(BaseTfIdfClassifier):
    '''
    The Tf-Idf SVM Classifier/Estimator.
    '''

    def __init__(self,
                 k: int = 6,
                 min_df: int = 2,
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False):
        super().__init__(k, min_df, use_idf, smooth_idf, sublinear_tf)

    def fit(self, X, y):
        check_classification_targets(y)
        self.num_classes_ = len(np.unique(y))
        self.estimator_ = Pipeline([('vec', TfidfVectorizer(lowercase=False,
                                                            analyzer='char',
                                                            encoding='latin1',
                                                            ngram_range=(self.k, self.k),
                                                            min_df=self.min_df,
                                                            use_idf=self.use_idf,
                                                            smooth_idf=self.smooth_idf,
                                                            sublinear_tf=self.sublinear_tf)),
                                    ('clf', LinearSVC(dual='auto', max_iter=5000))])
        self.estimator_.fit(X, y)
        return self
    
class TfIdfLogisticRegressionClassifier(BaseTfIdfClassifier):
    '''
    The Tf-Idf Logistic Regression Classifier/Estimator.
    '''

    def __init__(self,
                 k: int = 6,
                 min_df: int = 2,
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False):
        super().__init__(k, min_df, use_idf, smooth_idf, sublinear_tf)

    def fit(self, X, y):
        check_classification_targets(y)
        self.num_classes_ = len(np.unique(y))
        self.estimator_ = Pipeline([('vec', TfidfVectorizer(lowercase=False,
                                                            analyzer='char',
                                                            encoding='latin1',
                                                            ngram_range=(self.k, self.k),
                                                            min_df=self.min_df,
                                                            use_idf=self.use_idf,
                                                            smooth_idf=self.smooth_idf,
                                                            sublinear_tf=self.sublinear_tf)),
                                    ('clf', LogisticRegression(max_iter=5000))])
        self.estimator_.fit(X, y)
        return self
    
class TfIdfSvmAutonomousClassifier(BaseEstimator, ClassifierMixin):
    '''
    The Tf-Idf SVM Autonomous Classifier/Estimator.
    '''

    _parameter_constraints = {
        'valid_size' : [float]
    }

    def __init__(self,
                 valid_size : float = 0.5,
                 verbose: bool = True):
        self.valid_size = valid_size
        self.verbose = verbose

    def fit(self, X, y):
        check_classification_targets(y)
        self.num_classes_ = len(np.unique(y))

        # Select probable k values
        k_vals = choose_k_vals(len(set(list(X[1] + X[3] + X[5]))))

        if self.verbose:
            print('Searching for best paramters!')
        param_grid = {
                'vec__lowercase' : [False],
                'vec__analyzer' : ['char'],
                'vec__encoding' : ['latin1'],
                'vec__ngram_range' : [(k, k) for k in k_vals],
                'vec__min_df' : [2],
                'vec__sublinear_tf' : [False, True],
                'clf__dual' : ['auto'],
                'clf__max_iter' : [5000],
                'clf__C' : [0.1, 1, 10] }
        pipeline = Pipeline([('vec', TfidfVectorizer()),
                             ('clf', LinearSVC())])
        cv = StratifiedShuffleSplit(n_splits=1, train_size=self.valid_size, test_size=0.2*self.valid_size)
        gscv = GridSearchCV(pipeline, param_grid, cv=cv)
        gscv.fit(X, y)
        if self.verbose:
            print(f'Best Params, k: {gscv.best_params_["vec__ngram_range"][0]}, Sublinear TF: {gscv.best_params_["vec__sublinear_tf"]}, C: {gscv.best_params_["clf__C"]}')
        self.estimator_ = gscv.best_estimator_

        return self
    
    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)

class TfIdfLogisticRegressionAutonomousClassifier(BaseEstimator, ClassifierMixin):
    '''
    The Tf-Idf LogisticRegression Autonomous Classifier/Estimator.
    '''

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def fit(self, X, y):
        check_classification_targets(y)
        self.num_classes_ = len(np.unique(y))

        # Select probable k values
        k_vals = choose_k_vals(len(set(list(X[1] + X[3] + X[5]))))

        if self.verbose:
            print('Searching for best paramters!')
        param_grid = {
                'vec__lowercase' : [False],
                'vec__analyzer' : ['char'],
                'vec__encoding' : ['latin1'],
                'vec__ngram_range' : [(k, k) for k in k_vals],
                'vec__min_df' : [2],
                'vec__sublinear_tf' : [False, True],
                'clf__max_iter' : [5000],
                'clf__C' : [0.1, 1, 10] }
        pipeline = Pipeline([('vec', TfidfVectorizer()),
                             ('clf', LogisticRegression())])
        cv = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.1)
        gscv = GridSearchCV(pipeline, param_grid, cv=cv)
        gscv.fit(X, y)
        if self.verbose:
            print(f'Best Params, k: {gscv.best_params_["vec__ngram_range"][0]}, Sublinear TF: {gscv.best_params_["vec__sublinear_tf"]}, C: {gscv.best_params_["clf__C"]}')
        self.estimator_ = gscv.best_estimator_

        return self
    
    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)

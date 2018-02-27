



def get_as_matrix_wrapper_params(inner_model_params):
    return scope.get_as_matrix_wrapper(inner_model_params)



@scope.define
def get_as_matrix_wrapper(*args, **kwargs):
    return AsMatrixWrapper(*args, **kwargs)



class AsMatrixWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def fit(self, X, y):
        self._feature_names = np.array(list(X.columns.values))
        self.inner_model.fit(X.as_matrix(), y)
        return self

    def predict(self, X):
        return self.inner_model.predict(X.as_matrix())

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be true")
        return self._feature_names[self.inner_model.get_support(indices=True)]


###################################



def get_matrix_cleaning_wrapper_params(inner_model_params):
    return scope.get_matrix_cleaning_wrapper(inner_model_params)



@scope.define
def get_matrix_cleaning_wrapper(*args, **kwargs):
    return MatrixCleaningWrapper(*args, **kwargs)



class MatrixCleaningWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def _drop(self, X):
        X = X.drop(self._to_drop, axis=1, inplace=False)
        return X.as_matrix()

    def get_support(self, indices=False):
        if indices == False:
            raise KeyError("indices should be true")
        support = self.inner_model.get_support(indices=True)
        return np.array([self._features_invert_index[el] for el in support])

    def _set_dropped(self, X, to_drop):
        self._to_drop = to_drop
        self._features_invert_index = list()
        to_drop_set = set(to_drop)
        for el in X.columns.values:
            if el not in to_drop_set:
                self._features_invert_index.append(el)

    def fit(self, X, y):
        X = X.copy()
        X[X != 1] = 0

        ones_count = X.sum(axis=0)
        to_drop = ones_count[(ones_count <= 2) |
                  (ones_count >= (X.shape[0] / 3))].index
        self._set_dropped(X, to_drop)
        X = self._drop(X)
        print "cleaner fit", X.shape
        self.inner_model.fit(X, y)
        #self.feature_importances_ = self.inner_model.feature_importances_
        return self

    def predict(self, X):
        X = X.copy()
        X[X != 1] = 0
        X = self._drop(X)
        print "cleaner predict:", X.shape
        return self.inner_model.predict(X)




###################################


class SparseWrapper(BaseEstimator):
    def __init__(self, inner_model):
        self.inner_model = inner_model

    def _to_sparse(self, X):
        return csr_matrix(np.array(X))

    def get_support(self, *args, **kwargs):
        return self.inner_model.get_support(*args, **kwargs)

    def set_params(self, n_estimators=None, **params):
        super(SparseWrapper, self).set_params(**params)
        if n_estimators is not None:
            self.inner_model.set_params(n_estimators=n_estimators)

    def fit(self, X, y):
        X = self._to_sparse(X)
        #print "sparser", X.shape
        self.inner_model.fit(X, y)
        self.feature_importances_ = self.inner_model.feature_importances_
        return self

    def predict(self, X):
        X = self._to_sparse(X)
        return self.inner_model.predict(X)

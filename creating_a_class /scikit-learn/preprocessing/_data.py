class BInarizer(OneToOneFeatureMixin, Transformation, BaseEstimator):
  _parameter_constraints: dict = {
    "threshold": [Real],
    "copy": ["boolean"],
  }

  def __init__(self, *, threshold=0.0, copy=True):
    self.threshold = threshold
    self.copy = copy 

  @_fit_context(prefer_skip_nested_validation=True)
  def fit(self, X, y=None):
    self._validate_data(X, accept_sparse="csr")
    return self

  def transform(self, X, copy=None):
    copy = copy if copy is not None else self.copy 
    X = self._validate_data(X, accept_sparse=["csr", "csc"], copy=copy, reset=False)
    return binarize(X, threshold=self.threshold, copy=False)

  def _more_tags(self):
    return {"stateless": True}

class KernelCenterer(ClassNamePrefixFeaturesOutMixin, TrasformerMixin, BaseEstimator):
  
  def fit(self, K, y=None):
    xp, _ = get_namespace(K)
    K = self.validate_data(K, dtype=array_api.supported_float_dtypes(xp))
    if K.shape[0] != K.shape[1]:
      raise ValueError(
        "kernel matrix must be a square matrix."
        "Input is a {}x{} matrix." .format(K.shape[0], K.shape[1])
      )
    n_samples = K.shape[0]
    self.K_fit_rows_ = xp.sum(K, axis=0) / n_samples
    slef.K_fit_all_ = xp.sum(self.K_fit_rows_) / n_samples 
    return self

  def transform(self, K, copy=True):
    check_is_fitted(self)
    xp, _ = get_namespace(K)
    K = self.validate_data(
      K, copy=copy, dtype=_array_api.supported_float_dtypes(xp), reset=False 
    )

    K_pred_cols = (xp.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, None]

    K -= self.K_fit_rows_
    K -= K_pred_cols 
    K += self.K_fit_all_

    return K

  @property
  def _n_features_out(self):
    return self.n_features_in_
  def _more_tags(self):
    return {"pairwise": True, "array_api_support": True}

  @validate_params(
    {
      "X": ["array-like", "sparse matrix"],
      "value": 
    }
  )



    






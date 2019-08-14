import pandas as pd

from timeflux.core.node import Node


class Predict(Node):
    """ Predict the final classification step of a model on testing data.

    This node expects a scikit pipeline from input port 'model'. Once the model is
    received, the node is ready.
    Once ready, the node calls `predict` method of the model on the
    input data values.
    It returns the input DataArray with updated dimension 'target'.

    Attributes:
        i_model (Port): model input, expects meta.
        i (Port): default input, expects DataArray and meta.
        o (Port): default output, provides DataArray and meta.

    References:

        See the documentation of `sklearn.pipeline.Pipeline
        <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Preload the model from file.

    """

    def __init__(self):

        super().__init__()
        self._model = None

    def update(self):
        if 'pipeline' in self.i_model.meta:
            self._model = self.i_model.meta['pipeline']
            self.logger.info(f'Node Predict received the fitted model {self._model}')
        # When we have no available model or when we have not received data,
        # there is nothing to do
        if self._model is None or not self.i.ready():
            return

        self.o = self.i
        _X = self.o.data.values
        self.o.data.target.values = self._model['label'].inverse_transform(self._model['values'].predict(_X))


class PredictProba(Node):
    """ Estimate the probability of the final classification step of a model on testing data.

    This node expects a scikit pipeline from input port 'model'. Once the model is
    received, the node is ready.
    Once ready, the node calls `predict_proba` method of the model on the
    input data values.
    It returns a DataFrame with as many columns as the model has _classes and as many rows as
    the data has observations, with values are the probability of the class to be observed.

    Attributes:
        i_model (Port): model input, expects meta.
        i (Port): default input, expects DataArray and meta.
        o (Port): default output, provides DataArray and meta.

    References:

        See the documentation of `sklearn.pipeline.Pipeline
        <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Preload the model from file.
    """

    def __init__(self):

        super().__init__()
        self._model = None

    def update(self):
        if 'pipeline' in self.i_model.meta:
            self._model = self.i_model.meta['pipeline']

        # When we have no available model or when we have not received data,
        # there is nothing to do
        if self._model is None or not self.i.ready():
            return
        self.o.meta = self.i.meta
        _X = self.i.data.values
        _proba = self._model['values'].predict_proba(_X)
        classes_ = self._model['values'].classes_
        self.o.data = pd.DataFrame(data=_proba, columns=self._model['label'].inverse_transform(classes_))

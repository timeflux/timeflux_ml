import pandas as pd
import xarray as xr
from timeflux.core.node import Node


class Transform(Node):
    """ Apply pipeline transformations to the data.

    This node expects a scikit pipeline from input port 'model'. Once the model is
    received, the node is ready.
    Once the node ready, the node calls the transform method of the model on the.
    input data values and return the transformed data.

    Attributes:
        i_model (Port): model input, expects meta.
        i (Port): default input, expects DataFrame or DataArray and meta.
        o (Port): default output, provides DataFrame or DataArray and meta.

    Args:
        coords (dict|None): Dictionary-like container of coordinate arrays in case the data
                    is of type DataArray.
        set_columns ('all'|list|None): List of columns to be assigned to the output data in case the
                    data is of type DataFrame. If 'all', columns remain unchanged.
                    (only if the number of columns remains unchanged).
        set_index ('all'|'last'|None): Method to use to set the index to the output data in case the
                    data is of type DataFrame. If 'all', index remain unchanged. If 'last', the
                    transformed data should have only one row and the index is set to the last timestamp
                    of the input data.

    Example:

        In this example, we stream a respiratory signal, we accumulate data over a calibration period to feed a min-max scaler.
        Once the model is fitted, we apply the transformation to the signal.

        We choose the transformation `MinMaxScaler` from `sklearn.preprocessing`.

        In this case, there is no need for target labelling (``has_targets`` =`False`)
        and the input data is of type pandas.DataFrame.

        It results in:

        .. image:: /../../timeflux_ml/doc/static/image/fittransform_io1.svg
           :align: center

    Notes:

        This node works together with a Fit node that initializes and fits a sklearn Pipeline object with
        estimator steps, that is sent through port with suffix 'model'.

        Note also that the last step of the Pipeline should be a transformation, not a classifier
        (in which case, one should use node Predict).


    References:

        See the documentation of `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Since registry will soon be deprecated, model should be saved in the meta.

    """

    def __init__(self, coords=None, set_columns=None, set_index=None):

        self._model = None
        self._coords = coords
        self._set_columns = set_columns
        self._set_index = set_index

    def update(self):

        if 'pipeline' in self.i_model.meta:
            self._model = self.i_model.meta['pipeline']['values']

        # When we have no available model or when we have not received data,
        # there is nothing to do
        if self._model is None or not self.i.ready():
            return

        if isinstance(self.i.data, xr.DataArray):
            coords = dict(self.i.data.coords)
            coords.update(self._coords)
            self.o.data = xr.DataArray(data=self._model.transform(self.i.data.values),
                                       dims=self.i.data.dims,
                                       coords=coords)
        else:  # isinstance(self.i.data, pd.DataFrame)
            self.o.data = pd.DataFrame(data=self._model.transform(self.i.data.values))
            if self._set_index == 'all':
                self.o.data.index = self.i.data.index
            elif self._set_index == 'last':
                if len(self.o.data.index) != 1:
                    raise ValueError(f'Transformation returns a DataFrame with '
                                     f'length different from 1. '
                                     f'Hence, cannot set index to {self._set_index} timestamp.')
                self.o.data.index = [self.i.data.index[-1]]

            if self._set_columns == 'all':
                self.o.data.columns = self.i.data.columns
            elif isinstance(self._set_columns, list):
                if len(self.o.data.columns) != len(self._set_columns):
                    raise ValueError(f'Transformation returns a DataFrame with '
                                     f'{self.o.data.shape[1]} columns. '
                                     f'Hence, cannot set columns to {self._set_columns}.')
                self.o.data.index = [self.i.data.index[-1]]

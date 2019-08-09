import numpy as np
import pandas as pd

from functools import partial

from timeflux.core.node import Node
from timeflux.core.registry import Registry
import xarray as xr


class Transform(Node):
    """ Apply pipeline transformations to the data.

    This node loads a scikit pipeline previously saved in the Registry.
    When it receives data, it reshape it using the specified ``stack_method`` and calls the transform method.
    It returns the transformed data.

    Attributes:
        i (Port): default input, expects DataFrame and meta.
        o (Port): default output, provides DataFrame.


    Example:

        In this example, we stream a respiratory signal, we accumulate data over a calibration period to feed a min-max scaler.
        Once the model is fitted, we apply the transformation to the signal.

        We choose the transformation `MinMaxScaler` from `sklearn.preprocessing`.

        In this case, there is no need for buffering before (``receives_epoch`` = `False`), no need for target labelling (``has_targets`` =`False`)
        and the data should be concatenated vertically (``stack_method`` = `v_stack`).


        The corresponding graph is:

        .. literalinclude:: /../../timeflux_ml/test/graphs/fit_transform1.yaml
           :language: yaml


        The process is:

        - **silent**:

            - Fit node does nothing but waiting for an opening gate trigger matching ``event_begins`` (here `accumulation_begins`)  in the ``event_label`` (here `label`) column of the event input to start accumulating.
            - Transform returns nothing and waits for a model in the registry.

        - **accumulate**:

            - Fit node accumulates data in a buffer and wait for a closing gate trigger matching ``event_begins`` (here `accumulation_ends`) in the ``event_label`` (here `label`) column of the event input to start fitting.
            - Transform returns nothing and waits for a model in the registry.

        - **transform**:

            - Fit node is back to silence and waits for a new opening gate to recalibrate its model.
            - Transform node could yet load the fitted model, applies it to the data and returns tranformed data.

        It results in:

        .. image:: /../../timeflux_ml/doc/static/image/fittransform_io1.svg
           :align: center

        Where:

        - accumulation gate opens when events ports receives ::

                                                         label
            2018-11-19 10:57:04.948857320  accumulation_begins


        - accumulation gate closes when events ports receives ::

                                                         label
            2018-11-19 10:57:27.910938531    accumulation_ends

        - model stored in transform._model["values"] :

            {'scaler': MinMaxScaler(copy=True, feature_range=(0, 1))}

        - with parameters ::

            transform._model["values"].named_steps['scaler'].data_max_ =  array([814.49725342])

            transform._model["values"].named_steps['scaler'].data_min_ = array([736.10308838])


    Notes:

        This node works together with a Fit node that initializes a sklearn pipeline with transformation steps, accumulates data, feed the model and save it in the registry.
        Hence, for shapes compatibility purpose, the ``stack_method`` should be the same as the one specified in the Fit process.
        Furthermore, the last step of the Pipeline should be a transformation, not a classifier (in which case, use node Predict).
        Note that the modules are indeed imported dynamically in the pipeline but the dependencies must be satisfied in the environement.

    References:

        See the documentation of `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Since registry will soon be deprecated, model should be saved in the meta.

    """

    def __init__(self, coords=None, set_columns=None, set_index=None):
        """
         Args:
            set_columns (str|None): Whether or not the columns names of the input data should be transfered to the output data (only if the number of columns remains unchanged). Default: `True`.
            set_index (str|None): Whether or not the index of the input data should be transfered to the output data (only if the number of rows remains unchanged). Default: `False`.
        """
        self._model = None
        self._set_columns = set_columns
        self._set_index = set_index

    def update(self):

        if 'pipeline' in self.i_fit.meta:
            self._model = self.i_fit.meta['pipeline']['values']

        # When we have no available model or when we have not received data,
        # there is nothing to do
        if self._model is None or self.i.data is None or self.i.data.empty:
            return

        if isinstance(self.i.data, xr.Dataset):
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


import numpy as np
import pandas as pd

from functools import partial

from timeflux.core.node import Node
from timeflux.core.registry import Registry



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

    def __init__(self, stack_method, registry_key="fit_pipeline", set_columns=True, set_index=False):
        """
         Args:
            stack_method (string|int): Method to use for stacking ('vstack' to use `numpy.vstack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.vstack.html>`_ ;  'hstack' to use `numpy.hstack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html>`_ ; int (`0`, `1`, or `2`) to use `numpy.stack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.stack.html>`_ on the specified axis.
            registry_key (str): The key on which to load the fitted models.
            set_columns (bool): Whether or not the columns names of the input data should be transfered to the output data (only if the number of columns remains unchanged).
            set_index (bool): Whether or not the index of the input data should be transfered to the output data (only if the number of rows remains unchanged).

        """

        self._registry_key = registry_key
        if stack_method == "vstack":
            self._stack = np.vstack
        elif stack_method == "hstack":
            self._stack = np.hstack
        elif type(stack_method) == int:
            self._stack = partial(np.stack, axis=stack_method)
        self._stackable = None
        self._model = None
        self._set_columns = set_columns
        self._set_index = set_index

    def update(self):

        if (self._model is None) & (not hasattr(Registry, self._registry_key)) :
            self.o.data = None

        elif (self._model is None) & (hasattr(Registry, self._registry_key)):
            self._model = getattr(Registry, self._registry_key)

        if self._model is not None:
            if self.i.data is not None:
                self.o.meta = self.i.meta
                if not self.i.data.empty:

                    _X = self._stack([self.i.data.values])
                    self.o.data = pd.DataFrame(data = np.squeeze(self._model["values"].transform(_X)))
                    # TODO: Handle cases were transformations lead to a NDArray (N>2), using XArray.
                    if self._set_index:
                        self.o.data.index = self.i.data.index
                    if self._set_columns:
                        self.o.data.columns = self.i.data.columns

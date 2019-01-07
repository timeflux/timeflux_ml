
import numpy as np

from functools import partial

from timeflux.core.node import Node
from timeflux.core.registry import Registry



class Transform(Node):
    """ Apply pipeline transformations to the data.

    This node loads a scikit pipeline saved in the Registry.
    When it receives data, it reshape it using the specified ``stack_method`` and calls the transform method.
    It returns the transformed data.

    Attributes:
        i (Port): default input, expects DataFrame and meta.
        o (Port): default output, provides DataFrame.


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
            print(self._registry_key + " not in registry")
        elif (self._model is None) & (hasattr(Registry, self._registry_key)):
            self._model = getattr(Registry, self._registry_key)

        if self._model is not None:
            if self.i.data is not None:
                self.o = self.i
                if not self.i.data.empty:

                    _X = self._stack([self.i.data.values])
                    self.o.data = pd.DataFrame(data = np.squeeze(self._model["values"].transform(_X)))
                    # TODO: handle case were transformations lead to a NDArray (N>2), using XArray.
                    if self._set_index:
                        self.o.data.index = self.i.data.index
                    if self._set_columns:
                        self.o.data.columns = self.i.data.columns

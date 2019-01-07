
import numpy as np

from functools import partial

from timeflux.core.node import Node
from timeflux.core.registry import Registry



class Predict(Node):
    """   Applies final step of pipeline .

    This node loads a scikit pipeline saved in the Registry.
    When it receives data, it reshape it using the specified ``stack_method`` and calls the predict method.
    It adds a field in the meta with key ``meta_key`` and value the prediction.

    Attributes:
        i (Port): default input, expects DataFrame and meta.
        o (Port): default output, provides DataFrame and meta.


    References:

        See the documentation of `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Since registry will soon be deprecated, model should be saved in the meta.

    """

    def __init__(self, stack_method, registry_key="fit_pipeline", meta_key="pred" ):
        """
         Args:
            stack_method (string|int): Method to use for stacking ('vstack' to use `numpy.vstack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.vstack.html>`_ ;  'hstack' to use `numpy.hstack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html>`_ ; int (`0`, `1`, or `2`) to use `numpy.stack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.stack.html>`_ on the specified axis.
            registry_key (str): The key on which to load the fitted models.
            meta_key (str): The key to add in the output meta with the predicted label

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
        self._meta_key = meta_key

    def update(self):

        if (self._model is None) & (not hasattr(Registry, self._registry_key)) :
            self.o.data = None
        elif (self._model is None) & (hasattr(Registry, self._registry_key)):
            self._model = getattr(Registry, self._registry_key)
        if self._model is not None:
            if self.i.data is not None:
                self.o = self.i
                if not self.i.data.empty:
                    _X = self._stack([self.i.data.values])
                    # predict data label
                    if self.o.meta is None: self.o.meta={}
                    self.o.meta[self._meta_key] = self._model["label"].inverse_transform(self._model["values"].predict(_X))[0]


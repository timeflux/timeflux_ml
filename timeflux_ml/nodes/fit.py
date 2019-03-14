

import logging
import numpy as np
import pandas as pd
import json
from time import time

from itertools import cycle
from functools import partial
from importlib import import_module
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from timeflux.core.node import Node
from timeflux.core.registry import Registry


class Fit(Node):
    """ Pipeline of transforms with a final estimator.

    This node has 3 modes:

    - **silent**: do nothing but waiting for an opening gate trigger matching ``event_begins`` in the ``event_label`` column of the event input to change to mode `accumulate`
    - **accumulate**: accumulate data in a buffer and wait for a closing gate trigger matching ``event_ends`` in the ``event_label`` column of the event input to change to mode `fit`
    - **fit**: concatenate the buffer using method given in ``stack_method``, feed the model with the data, save it in the Registry and reset mode back to `silent`.

    The learning can be:

    - **supervised**:  if the training model requires to set `y` when calling the fit method, then data should be labeled in the meta, and ``has_targets`` set to `True` . Eventually, if ``context_key`` is given, the class of the data will be its value.
    - **unsupervised**: if the training model does not require to set `y` when calling the fit method, then ``has_targets`` should be set to `False`

    The model can handle:

    - **continuous data**: if the node is applied on streaming data, then ``receives_epochs`` should be `False`  (eg. scaling, pca, ... ).
    - **epochs**: if the node expects input data with strictky the same shape, then ``receives_epochs`` should be `True` (eg. XDawn, Covariances, ...).

    Attributes:
        i (Port): default input, expects DataFrame and meta.
        i_events (Port): event input, expects DataFrame.

    Notes:

        To assure the compatibility of labelling, we apply a LabelEncoder to the classes vector.

    References:

        See the documentation of `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Since registry will soon be deprecated, model should be saved in the meta.

    """

    def __init__(self, event_begins, event_ends, event_label, stack_method, steps_config, event_outputs_prefix="timeflux_fit", fit_params=None,  has_targets=True, receives_epochs=True, registry_key="fit_pipeline", context_key=None):
        """
        Args:
            event_begins (string): The marker name on which the node starts accumulating data.
            event_ends (string): The marker name on which the node stops accumulating data and fits the model.
            event_label (string): The column to match for event_trigger.
            event_outputs_prefix (string): The label prefixe of the output events stream.

            stack_method (string|int): Method to use for stacking ('vstack' to use `numpy.vstack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.vstack.html>`_ ;  'hstack' to use `numpy.hstack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html>`_ ; int (`0`, `1`, or `2`) to use `numpy.stack <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.stack.html>`_ on the specified axis.
            steps_config (list):  (name, module_name, method_name) Tuples to specify steps of the pipeline to fit.
            **fit_params (dict): string -> object.  Parameters passed to the fit method of
                                                    each step, where each parameter name is prefixed
                                                    such that parameter `p` for step `s` has key `s__p`.
            has_targets (bool, True): If True, model is supervised and meta should have a field "label"; if False, model is unsupervised and y is set to None. Default: `True` .
            receives_epochs (bool, True): if True, the node expects input data with strictly the same shapes. Default: `True` .
            registry_key (str): The key on which to save the fitted model. Default: `fit_pipeline`.
            context_key (str, None): The key on which the target is given. If None, the all "context" content is considered for label. Only if has_targets is True. Default: `None`.
        """

        self._event_begins = event_begins
        self._event_ends = event_ends
        self._event_label = event_label

        self._event_outputs_prefix = event_outputs_prefix

        if type(steps_config)==tuple: steps_config=[steps_config]
        self._steps_config = steps_config
        if fit_params is None: fit_params = {}
        self._fit_params = fit_params
        self._registry_key = registry_key
        self._has_targets = has_targets
        self._receives_epochs = receives_epochs
        self._context_key = context_key

        if stack_method == "vstack":
            self._stack = np.vstack
        elif stack_method == "hstack":
            self._stack = np.hstack
        elif type(stack_method) == int:
            self._stack = partial(np.stack, axis=stack_method)
        self._stackable = None

        self._reset()


    def _init_model(self):
        if not([len(steps) for steps in self._steps_config] == [3]*len(self._steps_config)):
            raise ValueError ("Parameter steps should be a list of (name, module_name, transform_name) tuples")

        self._steps = []
        for (name, transform_name, module_name) in self._steps_config:
            try:
                m = import_module(module_name)
            except ImportError:
                raise ImportError ("Could not import module {module_name}".format(module_name=module_name))
            try:
                transform = getattr(m, transform_name)()
            except AttributeError:
                raise ValueError ("Module {module_name} has no object {transform_name}".format(module_name=module_name, transform_name=transform_name))
            self._steps.append((name, transform))
        self._pipeline = Pipeline(steps=self._steps, memory=self._memory)
        try:
            self._pipeline.set_params(**self._fit_params)
        except ValueError:
            raise ValueError  ("Could not set params of pipeline. Check the validity. ")
        self._le = LabelEncoder()


    def _next(self):
        self._trigger = next(self._trigger_iterator)
        self._mode = next(self._mode_iterator)

    def update(self):

        # Detect onset to eventually update  mode
        if self.i_events.data is not None:
            if not self.i_events.data.empty:
                matches = self.i_events.data[self.i_events.data[
                    self._event_label] == self._trigger]
                if not matches.empty:
                    self._next()
                    if self.i.data is not None:
                        if not self.i.data.empty:
                            # troncate data to onset if not receiving epochs
                            if not self._receives_epochs:
                                if self._mode == "accumulate":
                                    self.i.data = self.i.data[matches.index[0]:]
                                elif self._mode == "fit":
                                    self.i.data = self.i.data[:matches.index[0]]
                                # TODO: handle case where begins & ends triggers are received at the same time

        # check mode
        if self._mode == "silent":
            # Do nothing
            pass

        elif self._mode in ["accumulate", "fit"] :
            # Append data
            if self._valid_input():
                # Append data
                self._buffer_values.append(self.i.data.values)

                # Append label
                if self._has_targets:
                    if self._context_key is not None:
                        if type(self.i.meta["epoch"]["context"]) == str : self.i.meta["epoch"]["context"] = json.loads(self.i.meta["epoch"]["context"])
                        self._buffer_label.append(self.i.meta["epoch"]["context"][self._context_key])
                    else:
                        self._buffer_label.append(self.i.meta["epoch"]["context"])

        if self._mode == "fit":


            self._fit()

            # Save in registry
            if self._has_targets:
                model = {"values": self._pipeline, "label": self._le}
            else:
                model = {"values": self._pipeline}

            setattr(Registry, self._registry_key, model)
            logging.info("Pipeline {registry_key} has been successfully saved in the registry".format(registry_key=self._registry_key))

            # Reset states
            self._reset()

    def _fit(self):

        # stack buffer
        _X = self._stack(self._buffer_values)

        if self._has_targets:
            _y = np.array(self._buffer_label)
            # Fit y model and save into the registery
            _y = self._le.fit_transform(_y)
        else:
            _y = None

        logging.debug("Wait for it, the model is fitting... ")
        _meta = {"X_shape": _X.shape}
        if _y is not None:
            _meta["y_count"] = Counter(set(_y))
        self.o_events.data = pd.DataFrame(index=[pd.Timestamp(time(), unit='s')],
                                          columns=["label", "data"],
                                          data=[[self._event_outputs_prefix + "_fitting-model_begins",
                                                 _meta]])

        # Fit X model
        self._pipeline.fit(_X, _y)

    def _valid_input(self):

        # Check input data and meta
        if self.i.data is not None:
            if not self.i.data.empty:

                if self._receives_epochs:
                    if self._shape is None:
                        self._shape = self.i.data.shape

                    # Check data shape
                    if self.i.data.shape != self._shape:
                        logging.warnings("FitPipeline received an epoch with invalid shape. Expecting {expected_shape}, "
                                         "received {actual_shape}.".format(expected_shape=self._shape, actual_shape=self.i.data.shape))
                        return False
                if self._has_targets:
                # Check valid meta is present
                    if (self.i.meta is None) | (self.i.meta is not None) & ("epoch" not in self.i.meta) |((self.i.meta is not None) & ("epoch" not in self.i.meta) & ("context" not in self.i.meta['epoch'])):
                        logging.warnings("FitPipeline received an epoch with no valid meta")
                        return False
                    elif self._context_key is not None :
                        if self._context_key not in self.i.meta["epoch"]["context"]:
                            logging.warnings("FitPipeline received an epoch with no valid meta: {context_key} not in meta['epoch']['context]".format(context_key=self._context_key))
                            return False

                if self._stackable is None:
                    # check if the features are stackable
                    try:
                        _ = self._stack([self.i.data.values, self.i.data.values])
                    except np.core._internal.AxisError:
                        raise ("Could not concatenate data. ")
                    self._stackable = True
                return True

    def _reset(self):

        # Reset buffers
        self._buffer_values = []
        if self._has_targets: self._buffer_label = []
        self._shape = None

        # Reset models
        self._le = LabelEncoder()
        self._init_model()

        # Reset iterator states
        self._trigger_iterator = cycle([self._event_begins, self._event_ends])
        self._mode_iterator = cycle(["silent", "accumulate", "fit"])
        self._trigger = next(self._trigger_iterator)
        self._mode = next(self._mode_iterator)


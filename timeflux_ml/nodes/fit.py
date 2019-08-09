import numpy as np
import pandas as pd
import json
from threading import Thread
from time import time, clock

from collections import Counter

from sklearn.preprocessing import LabelEncoder
import xarray as xr
from timeflux.core.node import Node

from copy import deepcopy

from timeflux_ml.utils.sklearn_helpers import construct_pipeline


class Fit(Node):
    """ Pipeline of transforms with a final estimator.

    This node has 3 modes:

    - **silent**: do nothing but waiting for an opening gate trigger matching ``event_begins`` in the ``event_label``
    column of the event input to change to mode `accumulate`
    - **accumulate**: accumulate data in a buffer and wait for a closing gate trigger matching ``event_ends`` in the
    ``event_label`` column of the event input to change to mode `fit`
    - **fit**: concatenate the buffer using method given in ``stack_method``, feed the model with the data, save it in
    the Registry and reset mode back to `silent`.

    The learning can be:

    - **supervised**:  if the training model requires to set `y` when calling the fit method, then data should be
    labeled in the meta, and ``has_targets`` set to `True` . Eventually, if ``context_key`` is given, the class of the
    data will be its value.
    - **unsupervised**: if the training model does not require to set `y` when calling the fit method, then
    ``has_targets`` should be set to `False`

    The model can handle:

    - **continuous data**: if the node is applied on streaming data, then ``receives_epochs`` should be `False`
    (eg. scaling, pca, ... ).
    - **epochs**: if the node expects input data with strictly the same shape, then ``receives_epochs`` should be `True`
    (eg. XDawn, Covariances, ...).

    Attributes:
        i (Port): default input, expects DataFrame and meta.
        i_events (Port): event input, expects DataFrame.

    Notes:

        To assure the compatibility of labelling, we apply a LabelEncoder to the classes vector.

    References:

        See the documentation of `sklearn.pipeline.Pipeline
        <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    **To do** :

        Since registry will soon be deprecated, model should be saved in the meta.

    """

    def __init__(self, pipeline_steps, pipeline_params,
                 event_label_base="timeflux_fit", has_targets=True):
        """
        Args:

            event_label_tx_base (string|None): The label prefixe of the output events stream.

            pipeline_steps (dict):  (name, module_name, method_name) Tuples to specify steps of the pipeline to fit.
            pipeline_params (dict): string -> object.  Parameters passed to the fit method of
                                                    each step, where each parameter name is prefixed
                                                    such that parameter `p` for step `s` has key `s__p`.
            has_targets (bool, True): If True, model is supervised and meta should have a field "label"; if False,
            model is unsupervised and y is set to None. Default: `True` .
            receives_epochs (bool, True): if True, the node expects input data with strictly the same shapes.
            Default: `True` .
            registry_key (str): The key on which to save the fitted model. Default: `fit_pipeline`.
            context_key (str, None): The key on which the target is given. If None, the all "context" content is
            considered for label. Only if has_targets is True. Default: `None`.
            allow_recalibration (bool):
        """

        self._pipeline_steps = pipeline_steps
        self._pipeline_params = pipeline_params or {}
        self._event_label_base = event_label_base

        self._has_targets = has_targets
        self._reset()
        self._thread = None

    def _reset(self):
        self._le = LabelEncoder()
        self._pipeline = construct_pipeline(self._pipeline_steps, self._pipeline_params)

    def update(self):
        super().update()

        # At this point, we are sure that we have some data to process
        self.o.data = pd.DataFrame()

        if self._thread is None:
            # When we have not received data, there is nothing to do
            if self.i.data is None or len(self.i.data == 0):
                return
            self._fit()

        if not self._thread.is_alive():
            # Save in registry
            if self._has_targets:
                model = {"values": deepcopy(self._pipeline), "label": deepcopy(self._le)}
            else:
                model = {"values": deepcopy(self._pipeline)}

            self.o.meta.update({'pipeline': model})

            self.logger.info(f'Pipeline {self._pipeline} was successfully fitted')

            # send an event to announce that fitting is ready.
            if self._event_label_base is not None:
                self.o.data = self.o.data.append(pd.DataFrame(index=[pd.Timestamp(time(), unit='s')],
                                                              columns=['label', 'data'],
                                                              data=[[self._event_label_base + '_ends',
                                                                     json.dumps(
                                                                         {'fit_duration': self._fit_duration})]]))

            self.logger.debug('Fit time: %.2f' % self._fit_duration)
            self._reset()

    def _fit(self):
        self._y = None
        _meta = {}
        if isinstance(self.i.data, xr.Dataset):
            self._X = self.i.data.data.values
            if self._has_targets:
                self._y = self.i.data.target.values
                self._y = self._le.fit_transform(self._y)
                y_count = dict(Counter(self._y))
                _meta['y_count'] = {self._le.inverse_transform([k])[0]: v for (k, v) in
                                    y_count.items()}  # convert keys to string and dump
        else:  # DataFrame or DataArray
            self._X = self.i.data.values

        _meta['X_shape'] = self._X.shape

        # save the models in the meta
        self.o.meta.update({'X': self._X, 'y': self._y})

        self.logger.info('Wait for it, the model is fitting... ')

        if self._event_label_base is not None:
            self.o.data = self.o.data.append(
                self.o.data.append(pd.DataFrame(index=[pd.Timestamp(time(), unit='s')],
                                                columns=['label', 'data'],
                                                data=[[self._event_label_base + '_begins', json.dumps(_meta)]])))

        # Fit X model in a thread
        self._thread = Thread(target=self._fit_pipeline)
        self._thread.start()

    def _fit_pipeline(self):
        tic = clock()
        self._pipeline.fit(self._X, self._y)
        self._fit_duration = clock() - tic

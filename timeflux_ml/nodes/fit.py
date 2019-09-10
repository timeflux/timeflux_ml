import numpy as np
import pandas as pd
import json
from threading import Thread
from time import time, clock

from collections import Counter

from sklearn.preprocessing import LabelEncoder
import xarray as xr
from timeflux.core.node import Node
from timeflux.core.exceptions import WorkerInterrupt

from copy import deepcopy

from timeflux_ml.utils.sklearn_helpers import make_pipeline


class Fit(Node):
    """ Construct and fit a sklearn Pipeline object

    This node first constructs a sklearn Pipeline object given ``pipeline_steps`` and
    ``pipeline_params``.
    Then, when data is received, the node calls the method `fit` of the pipeline object
    in a thread aside.
    Once the model has fitted, the node sends an event in data of output port with suffix 'events'
    and the fitted model in meta of output port with suffix 'model'.

    The fitting can be:

    - **supervised**:  if the training model requires to set `y` when calling the fit method,
        then parameter ``has_targets`` set to `True` and the data should be od type DataArray with
        dimension `target`.
    - **unsupervised**: if the training model does not require to set `y` when calling the fit method,
        then parameter ``has_targets`` should be set to `False` and data can be either of type DataFrame
        or DataArray.

    Attributes:
        i (Port): default input, expects DataFrame or DataArray.
        o_events (Port): event output, provides DataFrame.
        o_model (Port): model output, provides meta.

    Args:
        pipeline_steps (dict):  string -> string. Keys are step names and values are
            estimator class names (eg. {'scaler': 'sklearn.preprocessing.MinMaxScaler'})
        pipeline_params (dict): string -> object. Parameters passed to the fit method of
            each step, where each parameter name is prefixed such that parameter `p` for
            step `s` has key `s__p`. (eg. {'scaler__feature_range': (.1, .99)})
        event_label_base (string|None): The label prefix of the output events stream to
            inform that model starts/ends fitting.
        has_targets (bool, True): If True, model is supervised and meta should have a
            field "label"; if False,


    Notes:
        In case the fitting is of type `supervised`, we systematically apply a LabelEncoder
        to the classes vector  in order to assure the compatibility of labelling,

        Note also that the modules are indeed imported dynamically in the pipeline
        but the dependencies must be satisfied in the environment.

    References:

        See the documentation of `sklearn.pipeline.Pipeline
        <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_

    """

    def __init__(self, pipeline_steps, pipeline_params=None,
                       event_label_base='model-fitting', has_targets=True):

        super().__init__()
        self._pipeline_steps = pipeline_steps
        self._pipeline_params = pipeline_params or {}
        self._event_label_base = event_label_base

        self._has_targets = has_targets
        self._reset()
        self._thread = None
        self._thread_status = None

        # todo: preload model from file.

    def _reset(self):
        self._le = LabelEncoder()
        self._pipeline = make_pipeline(self._pipeline_steps, self._pipeline_params)
        self._thread = None
        self._thread_status = None

    def update(self):

        # At this point, we are sure that we have some data to process
        self.o_events.data = pd.DataFrame()

        if self._thread_status is None:
            # When we have not received data, there is nothing to do
            if not self.i.ready():
                return
            self._fit()

        if self._thread_status == 'FAILED':
            raise WorkerInterrupt('Estimator fit failed.')

        elif self._thread_status == 'SUCCESS':
            if self._has_targets:
                model = {'values': deepcopy(self._pipeline), 'label': deepcopy(self._le)}
            else:
                model = {'values': deepcopy(self._pipeline)}

            self.o_model.meta.update({'pipeline': model})

            self.logger.info(f'The model {self._pipeline} was successfully fitted. ')

            # send an event to announce that fitting is ready.
            if self._event_label_base is not None:
                self.o_events.data = self.o_events.data.append(pd.DataFrame(index=[pd.Timestamp(time(), unit='s')],
                                                                            columns=['label', 'data'],
                                                                            data=[[self._event_label_base + '_ends',
                                                                                   '']]))

            self._reset()
        else:  # self._thread_status == 'WORKING'
            return

    def _fit(self):
        self._thread_status = 'WORKING'
        self._y = None
        _meta = {}
        self._X = self.i.data.values
        if self._has_targets:
            if isinstance(self.i.data, xr.DataArray):
                self._y = self.i.data.target.values
                # self._y = self.i.data.target.values
                self._y = self._le.fit_transform(self._y)
                y_count = dict(Counter(self._y))
                _meta['y_count'] = {self._le.inverse_transform([k])[0]: v for (k, v) in
                                    y_count.items()}  # convert keys to string and dump
            else:
                raise ValueError('If `has_target` is True, the node expects '
                                 'a DataArray with dimension "target"')

        _meta['X_shape'] = self._X.shape

        # save the models in the meta
        self.o_model.meta.update({'X': self._X, 'y': self._y})

        self.logger.info('Please wait, the model is fitting... ')

        if self._event_label_base is not None:
            self.o_events.data = pd.DataFrame(index=[pd.Timestamp(time(), unit='s')],
                                              columns=['label', 'data'],
                                              data=[[self._event_label_base + '_begins', json.dumps(_meta)]])

        # Fit X model in a thread
        self._thread = Thread(target=self._fit_thread)
        self._thread.start()

    def _fit_thread(self):
        try:
            self._pipeline.fit(self._X, self._y)
            self._thread_status = 'SUCCESS'
        except ValueError:
            self._thread_status = 'FAILED'

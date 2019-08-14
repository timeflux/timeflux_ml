from collections import Counter
from threading import Thread
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from timeflux.core.exceptions import WorkerInterrupt
from timeflux.core.node import Node
from timeflux_ml.utils.import_helpers import make_object
from timeflux_ml.utils.sklearn_helpers import make_pipeline


class CrossValidate(Node):
    """ Evaluate metric(s) by cross-validation on the fitted model and data.

    This node first constructs a sklearn Pipeline object given ``pipeline_steps`` and
    ``pipeline_params`` and a sklearn CV given ``cv`` and ``cv_params``.
    Then, when data is received, the node cross-validates the model on the data and returns
    an event with the resulting scores.

    When it receives data, it calls the `cross_validate` method from `sklearn.model_selection` and returns an event
    with label ``event_label`` and data, a dict of arrays containing the average score/time arrays for each scorer.

    Attributes:
        i (Port): default input, expects DataArray.
        o_events (Port): events output, provides DataFrame.

    References:

        See the documentation of `sklearn.model_selection.cross_validate
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>`_

    **To do** :

        Load data and model from different ports to allow for transfer-learning validation.

    """

    def __init__(self, pipeline_steps, pipeline_params=None, event_label='model-cv_reports',
                 cv='sklearn.model_selection.StratifiedKFold', cv_params=None, scoring=None):

        """
         Args:
            module_name (str): name of the module to import, in which the cv is defined. Default: `sklearn.model_selection`.
            cv_name (str): name of the BaseCrossValidator that determines the cross-validation splitting strategy. . Default: `StratifiedKFold`.
            cv_kwargs (dict): Keyword Arguments to pass to the BaseCrossValidator.
            scoring (str|dict|list|None): A single string (see The scoring parameter: defining model evaluation rules) or a callable to evaluate the predictions on the test set. If None, the estimator’s default scorer (if available) is used.
            registry_key (str): The key on which to save the fitted model. Default: `fit_pipeline`.
            event_label_tx_base (string): The label prefixe of the output events stream.

        """

        self._pipeline_steps = pipeline_steps
        self._pipeline_params = pipeline_params or {}

        cv_params = cv_params or {}
        self._cv = make_object(cv, cv_params)
        self._scoring = scoring or {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}
        self._event_label = event_label

        self._reset()
        self._thread = None
        self._thread_status = None

    def _reset(self):
        self._event_data = {}
        self._le = LabelEncoder()
        self._pipeline = make_pipeline(self._pipeline_steps, self._pipeline_params)
        self._thread = None
        self._thread_status = None

    def update(self):

        # At this point, we are sure that we have some data to process

        if self._thread_status is None:
            # When we have not received data, there is nothing to do
            if not self.i.ready():
                return
            self._cross_validate()

        if self._thread_status == 'FAILED':
            raise WorkerInterrupt('Estimator fit failed.')

        elif self._thread_status == 'SUCCESS':
            self.logger.info(f'Scores from cross-validation are: {self._scores}. ')

            # send an event to announce that fitting is ready.
            self.o_events.data = pd.DataFrame(index=[pd.Timestamp(time(), unit='s')],
                                                  columns=['label', 'data'],
                                                  data=[[self._event_label,
                                                         self._scores]])

            self._reset()
        else:  # self._thread_status == 'WORKING'
            return

    def _cross_validate(self):
        self._thread_status = 'WORKING'
        self._X = self.i.data.values
        self._y = self.i.data.target.values
        # self._y = self.i.data.target.values
        self._y = self._le.fit_transform(self._y)
        y_count = dict(Counter(self._y))
        self._event_data['y_count'] = {self._le.inverse_transform([k])[0]: v for (k, v) in
                                       y_count.items()}  # convert keys to string and dump
        self._event_data['X_shape'] = self._X.shape

        # save the models in the meta
        self._event_data.update({'X': self._X, 'y': self._y})

        self.logger.info('Please wait, the model is cross-validating... ')

        # Cross-validate model against X,y in a thread
        self._thread = Thread(target=self._cross_validate_thread)
        self._thread.start()

    def _cross_validate_thread(self):
        try:
            _scores = cross_validate(self._pipeline, self._X, self._y,
                                     cv=self._cv, scoring=self._scoring,
                                     verbose=False, return_train_score=True)
            self._scores = {k: np.mean(v) for (k, v) in _scores.items()}

            self._thread_status = 'SUCCESS'
        except ValueError:
            self._thread_status = 'FAILED'

import json

import numpy as np
import pandas as pd
import pytest
from timeflux.helpers.testing import DummyData
from timeflux_ml.nodes.fit import Fit
from timeflux_ml.nodes.transform import Transform

# ------------------------------------------------------
# Use-case #1 : Not supervised fit Pipeline on pandas data
# ------------------------------------------------------
# params:
#    has_targets = False
#
# Inputs: DataFrame
# Outputs: DataFrame
#
# eg: MinMax scaler calibrated on streaming data.

num_cols = 5
data = DummyData(rate=10, jitter=.05, num_cols=num_cols)
node_fit = Fit(
    pipeline_steps={'scaler': 'sklearn.preprocessing.MinMaxScaler'},
    has_targets=False)

def test_fit_no_data():
    """Fit received empty DataFrame no data"""
    node_fit.i.data = pd.DataFrame()
    node_fit.update()
    assert node_fit.o.meta == {}

def test_fit_data():
    """Fit received some data and fit pipeline in a thread """
    calibration_size = 30
    output_data = pd.DataFrame()
    node_fit.i.data = data.next(calibration_size)
    while True:
        node_fit.update()
        output_data = output_data.append(node_fit.o_events.data)
        if 'pipeline' in node_fit.o_model.meta:
            break

    # assert model has fit on data between time_begins and time_ends
    np.testing.assert_array_equal(node_fit.o_model.meta['pipeline']['values'].named_steps['scaler'].data_max_,
                                  data._data.iloc[:calibration_size].max().values,
                                  [0.962992, 0.984083, 0.939068, 0.991169, 0.997934])
    np.testing.assert_array_equal(node_fit.o_model.meta['pipeline']['values'].named_steps['scaler'].data_min_,
                                  data._data.iloc[:calibration_size].min().values,
                                  [0.037008, 0.086987, 0.024401, 0.006386, 0.021269])
    np.testing.assert_array_equal(node_fit.o_model.meta['pipeline']['values'].named_steps['scaler'].data_range_,
                                  data._data.iloc[:calibration_size].apply(lambda x: np.ptp(x)).values,
                                  [0.925984, 0.897096, 0.914667, 0.984783, 0.976665])

    expected_data = np.array([['model-fitting_begins', json.dumps({'X_shape': [calibration_size, num_cols]})],
                              ['model-fitting_ends', '']])

    np.testing.assert_array_equal(output_data.values, expected_data)

def test_transform_data():
    """ Transform data
    Plug output model from fit to a Transform node and apply transformation to a chunk of data"""
    node_transform = Transform(set_columns='all', set_index='all')

    node_transform.i.data = data.next(3)
    node_transform.i_model = node_fit.o_model
    node_transform.update()

    expected_transformed = np.array([[0.74123574, 0.83273212, 0.73186989, 0.6804381, 0.61161401],
                                     [0.79020651, 0.07951539, 0.93129185, 0.87740099, 0.00816861],
                                     [0.87091542, 0.04650148, 0.33786255, 0.74554375, 0.14282379]])

    np.testing.assert_array_almost_equal(node_transform.o.data.values, expected_transformed, 6)

def test_transform_invalid_parameter():
    """ Invalid parameter `set_index` for transformed data"""
    node_transform_invalid = Transform(set_columns='all', set_index='last')
    node_transform_invalid.i.data = data.next(3)
    node_transform_invalid.i_model = node_fit.o_model
    with pytest.raises(ValueError):
        node_transform_invalid.update()

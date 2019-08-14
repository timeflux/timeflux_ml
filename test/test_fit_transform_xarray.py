import pandas as pd
import numpy as np
import xarray as xr

from timeflux_ml.nodes.fit import Fit
from timeflux_ml.nodes.transform import Transform

# ------------------------------------------------------
# Use-case #2 : Supervised fit Pipeline on xarray data
# ------------------------------------------------------
# params:
#    has_targets = True
#
# Inputs: DataArray
# Outputs: DataArray
#
# eg: fit a XDawn spatial filter to discriminate two classes
#

np.random.seed(0)

n_trials = 100  # nb of trials
n_channels = 5
n_times = 10
n_filters = 2
n_classes = 2  # 'foo' and 'bar'

node_fit = Fit(
    pipeline_steps={'xdawn': 'pyriemann.spatialfilters.Xdawn'},
    pipeline_params={'xdawn__nfilter': n_filters},
    has_targets=True)
target = np.array(['foo', 'bar']).repeat(50)

data = xr.DataArray(np.random.randn(n_trials, n_channels, n_times), dims=('target', 'space', 'time'),
                    coords=(target, [f'ch{k}' for k in range(n_channels)],
                            pd.date_range(
                                start='2018-01-01',
                                periods=n_times,
                                freq=pd.DateOffset(seconds=100))))


def test_fit_data():
    """Fit received some data and fit pipeline in a thread """

    node_fit.i.data = data
    output_data = pd.DataFrame()
    while True:
        node_fit.update()
        output_data = output_data.append(node_fit.o_events.data)
        if 'pipeline' in node_fit.o_model.meta:
            break

    expected_filters = np.array([[-0.54364819, -0.60638953, -0.36703277, -0.34184081, 0.29183931],
                                 [0.20890224, -0.34419412, 0.48460301, -0.6253412, -0.46043302],
                                 [-0.76839534, 0.42929239, 0.3013263, 0.28496854, -0.23080738],
                                 [0.09069146, 0.13413278, -0.7122559, 0.65923861, -0.17854816]])
    np.testing.assert_array_almost_equal(node_fit.o_model.meta['pipeline']['values'].named_steps['xdawn'].filters_,
                                         expected_filters, 6)


def test_transform_data():
    """ Transform data
    Plug output model from fit to a Transform node and apply transformation to a chunk of data"""

    node_transform = Transform(coords={'space': [f'new{k}' for k in range(n_classes * n_filters)]})
    node_transform.i_model = node_fit.o_model
    node_transform.i.data = data[[0], :, :]
    node_transform.update()

    expected_transform_epoch = np.array([[[-0.468316, -1.882989, -1.505326, 0.226789, -2.147329, 0.681465,
                                           -2.225481, 0.091273, -1.034901, -0.203356],
                                          [-0.532358, 0.317129, 1.702216, 0.407065, 1.789516, -0.919862,
                                           -0.486126, -1.161594, 1.598774, 1.3788],
                                          [-1.776767, 0.949297, -0.024044, -2.90801, -0.542052, 0.601606,
                                           0.56489, 0.135016, 0.937701, -0.276447],
                                          [2.287052, 0.268654, -0.705468, -0.905963, -1.526098, 1.173295,
                                           1.288718, 0.745913, -1.026282, -1.285189]]])
    np.testing.assert_almost_equal(node_transform.o.data.values, expected_transform_epoch, 6)

import pandas as pd
import numpy as np
import pytest
# import helpers
from . import helpers
from timeflux.nodes.epoch import Epoch
from timeflux.core.registry import Registry
from timeflux_ml.nodes.fit import Fit
from timeflux_ml.nodes.transform import Transform
from timeflux_ml.nodes.predict import Predict

from time import sleep

#------------------------------------------------------
# Check-up testing : Test invalid syntax/inputs
#------------------------------------------------------
def test_invalid_input():
    # test wrong syntax for module/method
    with pytest.raises(ValueError):
        node = Fit(event_begins="accumulation_begins", event_ends="accumulation_ends", event_label="label",
                   stack_method="vstack",
                   steps_config=("scaler", "MinMaxScalerr", "sklearn.preprocessing"),
                   fit_params=None,
                   has_targets=False,
                   receives_epochs=False,
                   registry_key="test_scaler")

    # test invalid syntax for params
    with pytest.raises(ValueError):
        node = Fit(event_begins="accumulation_begins", event_ends="accumulation_ends", event_label="label",
                   stack_method="vstack",
                   steps_config=("scaler", "MinMaxScaler", "sklearn.preprocessing"),
                   fit_params={"scaler__foo": "bar"},
                   has_targets=False,
                   receives_epochs=False,
                   registry_key="test_scaler")


def test_not_stackable():
    # test stack method is invalid
    data = helpers.DummyData()
    node = Fit(event_begins="accumulation_begins", event_ends="accumulation_ends", event_label="label",
               stack_method=3,
               steps_config=("scaler", "MinMaxScaler", "sklearn.preprocessing"),
               fit_params=None,
               has_targets=False,
               receives_epochs=False,
               registry_key="test_scaler")
    node.i.data = data.next(10)
    node.i_events.data = pd.DataFrame([['accumulation_begins']], [node.i.data.index[0]], columns=['label'])
    with pytest.raises((ValueError, TypeError)):
        node.update()





#------------------------------------------------------
# Use-case #1 : Fit Pipeline on streaming data.
#------------------------------------------------------
# params:
#    has_targets = False
#    receives_epochs = False
#
# eg: MinMax scaler calibrated on streaming data.
#

backup_data = helpers.DummyData( rate=10, jitter=.05,)
all_data = backup_data.next(50)


data = helpers.DummyData()
node_fit1 = Fit(event_begins="accumulation_begins", event_ends="accumulation_ends", event_label="label",
           stack_method="vstack",
           steps_config=("scaler", "MinMaxScaler", "sklearn.preprocessing"),
           fit_params=None,
           has_targets=False,
           receives_epochs=False,
           registry_key="test_scaler")

node_transform = Transform( stack_method="vstack",
                               registry_key="test_scaler")

time_begins = pd.Timestamp("2018-01-01 00:00:00.496559945")
time_ends = pd.Timestamp("2018-01-01 00:00:03.801842330")

def test_start_accumulating():
    # send a chunk data with 10 rows and an event that triggers the begining of the data accumulation
    node_fit1.i.data = data.next(10)
    event = pd.DataFrame([['accumulation_begins']], [time_begins], columns=['label'])  # Generate a trigger event
    node_fit1.i_events.data = event

    # mode should be silent
    assert node_fit1._mode == 'silent'

    node_fit1.update()

    # mode should be accumulating
    assert node_fit1._mode == 'accumulate'

    # buffer_values should contain one chunk of data corresponding to node.i.data.iloc[5:]
    expected_buffer_values = np.array([[0.474214, 0.862043, 0.844549, 0.3191  , 0.828915],
                                       [0.037008, 0.59627 , 0.230009, 0.120567, 0.076953],
                                       [0.696289, 0.339875, 0.724767, 0.065356, 0.31529 ],
                                       [0.539491, 0.790723, 0.318753, 0.625891, 0.885978],
                                       [0.615863, 0.232959, 0.024401, 0.870099, 0.021269]])
    np.testing.assert_array_equal(expected_buffer_values, node_fit1._buffer_values[0])



def test_fit_model_1():
    node_fit1.i.data = data.next(30)
    event = pd.DataFrame([['accumulation_ends']], [time_ends], columns=['label'])  # Generate a trigger event
    node_fit1.i_events.data = event
    # node_fit1.update()
    while ((node_fit1._thread) is None or (not node_fit1._thread.isAlive())) and (node_fit1._mode!="silent"):
        node_fit1.update()
        sleep(0.1)

    # assert model is saved in registry
    assert hasattr(Registry, "test_scaler") == True

    # assert model has fit on data between time_begins and time_ends
    np.testing.assert_array_equal(Registry.test_scaler["values"].named_steps["scaler"].data_max_,
                                  all_data.loc[time_begins:time_ends].max().values,
                                  [0.962992, 0.984083, 0.939068, 0.991169, 0.997934])
    np.testing.assert_array_equal(Registry.test_scaler["values"].named_steps["scaler"].data_range_,
                                  all_data.loc[time_begins:time_ends].apply(np.ptp).values,
                                  [0.925984, 0.897096, 0.914667, 0.984783, 0.976665])
    np.testing.assert_array_equal(Registry.test_scaler["values"].named_steps["scaler"].data_min_,
                                  all_data.loc[time_begins:time_ends].min().values,
                                  [0.037008, 0.086987, 0.024401, 0.006386, 0.021269])

    # mode should be silent and waiting for a "accumulation_begins" trigger
    assert node_fit1._mode == 'silent'
    assert node_fit1._trigger == 'accumulation_begins'


def test_transform_1():
    # test transforming next chunk of data based on fitted model
    node_transform = Transform( stack_method="vstack",
                               registry_key="test_scaler")

    # test transforming next chunk of data based on fitted model

    node_transform.i.data = data.next(3)
    node_transform.update()

    expected_transformed = np.array([[0.77890655, -0.06955777, -0.00248943, 0.32212579, 0.47854075],
                                     [0.79202124, 0.66470924, 0.46082563, 0.27137044, 0.99917167],
                                     [0.42028048, 0.40619956, 0.15221168, 0.80060683, 0.68847865]])
    np.testing.assert_array_almost_equal(node_transform.o.data.values, expected_transformed, 6)

#------------------------------------------------------
# Use-case #2 : Fit Pipeline on labelled epochs data.
#------------------------------------------------------
# params:
#    has_targets = True
#    receives_epochs = True
#    stack_method = 0
#
# eg: fit a XDawn spatial filter to discriminate two classes
#
n_trials = 12  # nb of trials
n_channels = 5
n_samples = 10
n_filters = 2
n_classes = 2  # 'foo' and 'bar'

context = ["foo"] * int(n_trials / 2) + ["bar"] * int(n_trials / 2)

def test_fit_model_2():


    backup_data = helpers.DummyData(rate=10, jitter=0., num_cols=n_channels)
    all_data = backup_data.next(50)

    data = helpers.DummyData(rate=10, jitter=0., num_cols=n_channels)
    node_epoch = Epoch(event_trigger='fit_xdawn', before=0.2, after=0.7)
    node_fit2 = Fit(event_begins="accumulation_begins", event_ends="accumulation_ends", event_label="label",
                   stack_method=0,
                   steps_config=("xdawn", "Xdawn", "pyriemann.spatialfilters"),
                   fit_params={"xdawn__nfilter": n_filters},
                   has_targets=True,
                   receives_epochs=True,
                   registry_key="pipe_xdawn")

    # send a trigger to start accumlulation
    event = pd.DataFrame(['accumulation_begins'], [all_data.index[0]], columns=['label'])
    node_fit2.i_events.data = event
    node_fit2.update()
    assert node_fit2._mode == "accumulate"
    assert node_fit2._trigger == "accumulation_ends"


    node_epoch.i_events.data = event
    for k in range(n_trials):
        node_epoch.i.data = data.next(n_samples + 1)
        time = node_epoch.i.data.index[2]   #Timestamp('2018-01-01 00:00:00.200000')
        event = pd.DataFrame([['fit_xdawn', context[k]]], [time], columns=['label', 'data']) # Generate a trigger event
        node_epoch.i_events.data = event
        node_epoch.update()
        node_fit2.i.meta = node_epoch.o.meta
        node_fit2.i.data = node_epoch.o.data.T
        node_fit2.update()

    _X = node_fit2._stack(node_fit2._buffer_values)
    assert _X.shape == (n_trials, n_channels, n_samples)
    _y = node_fit2._le.fit_transform(np.array(node_fit2._buffer_label))
    assert _y.shape == (n_trials,)

    np.testing.assert_array_equal(node_fit2._buffer_label, context)

    # send a trigger to stop accumlulation
    time_ends = data.next(1).index  # DatetimeIndex(['2018-01-01 00:00:11'], dtype='datetime64[ns])
    event = pd.DataFrame(['accumulation_ends'], [time_ends], columns=['label'])
    node_fit2.i_events.data = event

    while ((node_fit2._thread) is None or (not node_fit2._thread.isAlive())) and (node_fit2._mode!="silent"):

        node_fit2.update()
        sleep(0.1)

    # check the model has been saved in the Registry
    assert hasattr(Registry, "pipe_xdawn") == True

    # check the label encoder is correctly inversablle
    np.testing.assert_array_equal(Registry.pipe_xdawn["label"].inverse_transform(_y), context)

    # check the model has been correctly fitted
    Registry.pipe_xdawn["values"].named_steps["xdawn"]

    expected_filters_ = np.array([[ 0.44488995, -0.60825588,  0.20863747,  0.18988823,  0.59372601],
                                  [ 0.19963358, -0.60664856, -0.28307248, -0.15757341, -0.69797174],
                                  [-0.58033272, -0.48413254,  0.00169122,  0.59757889,  0.26781753],
                                  [-0.1991946 , -0.16305946,  0.40353354,  0.20193178, -0.85446905]])
    np.testing.assert_array_almost_equal( Registry.pipe_xdawn["values"].named_steps["xdawn"].filters_ , expected_filters_, 6)

    assert Registry.pipe_xdawn["values"].named_steps["xdawn"].transform(_X).shape == (n_trials, n_filters * n_classes, n_samples)

def test_transform_2():
    data = helpers.DummyData(rate=10, jitter=0., num_cols=n_channels)
    # test transforming next chunk of data based on fitted model
    node_epoch = Epoch(event_trigger='transform_xdawn', before=0.2, after=0.7)
    node_transform = Transform( stack_method=0,
                               registry_key="pipe_xdawn", set_columns=True)

    # test transforming next chunk of data based on fitted model
    node_epoch.i.data = data.next(n_samples + 1)
    time = node_epoch.i.data.index[2]  # Timestamp('2018-01-01 00:00:00.200000')

    event = pd.DataFrame([['transform_xdawn', None]], [time], columns=['label', 'data'])  # Generate a trigger event
    node_epoch.i_events.data = event
    node_epoch.update()
    node_transform.i.meta = node_epoch.o.meta
    node_transform.i.data = node_epoch.o.data.T
    node_transform.update()
    expected_transformed = np.array([[ 0.55279574,  0.38717185,  0.12102983,  0.61428735,  1.06730228,
         0.41557662, -0.2296485 ,  0.45386077,  0.47043355,  0.31523305],
       [-1.21722766, -0.90978351, -1.3209278 , -0.2535545 , -1.11491286,
        -1.2961976 , -0.49215669, -0.50270548, -1.17923204, -0.17723418],
       [ 0.28525908, -0.43575805, -0.39241433, -0.06724576,  0.28604614,
        -0.27843126, -0.21710401, -0.44390254, -0.08406024,  0.05550179],
       [-0.31429851, -0.26921188, -0.41908322,  0.1712963 , -0.42514168,
        -0.53806707, -0.05319056, -0.15785715, -0.73842454,  0.0067108 ]])

    np.testing.assert_array_almost_equal(expected_transformed, node_transform.o.data.values )

    np.testing.assert_array_equal(node_transform.o.data.columns, node_transform.i.data.columns)

#------------------------------------------------------
# Use-case #3 : Fit Pipeline on unlabelled epochs data.
#------------------------------------------------------
# params:
#    has_targets = False
#    receives_epochs = True
#
# eg: fit_predict
#

from sklearn import datasets
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# Load the Iris dataset
iris = datasets.load_iris()

pipeline = Pipeline([
      ('feature_selection', SelectKBest(k=2)),
      ('classification', RandomForestClassifier())
    ])

pipeline.fit(iris.data, iris.target)
prediction1 = pipeline.predict(iris.data)

pipeline2 = Pipeline([
      ('feature_selection', SelectKBest(k=2)),
      ('classification', RandomForestClassifier())
    ])

pipeline2.fit(iris.data, iris.target)
prediction2 = pipeline2.predict(iris.data)
node_fit3 = Fit(event_begins="accumulation_begins", event_ends="accumulation_ends", event_label="label",
                stack_method="vstack",
                steps_config=[("feature_selection", "SelectKBest", "sklearn.feature_selection"),
                              ("classification", "RandomForestClassifier", "sklearn.ensemble")],
                fit_params={"feature_selection__k": 2},
                has_targets=True,
                receives_epochs=True,
                registry_key="test_forest",
                context_key=None)
node_predict = Predict(stack_method="vstack", registry_key="test_forest", meta_key="pred")

def test_fit_3():
    event = pd.DataFrame([['accumulation_begins']], [0], columns=['label'])  # Generate a trigger event
    node_fit3.i_events.data = event
    node_fit3.update()

    for i_data, i_label in zip(iris.data, iris.target):
        node_fit3.i.data = pd.DataFrame(i_data).T
        node_fit3.i.meta = {"epoch": {"context": i_label}}
        node_fit3.update()

    node_fit3.i.data = None
    node_fit3.i.meta = None
    event = pd.DataFrame([['accumulation_ends']], [0], columns=['label'])  # Generate a trigger event
    node_fit3.i_events.data = event
    while ((node_fit3._thread) is None or (not node_fit3._thread.isAlive())) and (node_fit3._mode!="silent"):
        node_fit3.update()
        sleep(0.1)

    # # assert model is saved in registry
    assert hasattr(Registry, "test_forest") == True


def test_predict():
    from_registry_expected_prediction = Registry.test_forest["values"].predict(iris.data)
    from_registry_online_prediction = []
    for i_data, i_label in zip(iris.data, iris.target):
        node_predict.i.data = pd.DataFrame(i_data).T
        node_predict.i.meta = {"epoch": {"context": i_label}}
        node_predict.update()
        from_registry_online_prediction.append(node_predict.o.meta["pred"])

    np.testing.assert_array_equal(from_registry_expected_prediction, from_registry_online_prediction)



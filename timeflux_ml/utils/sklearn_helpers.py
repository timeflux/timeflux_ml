from sklearn.pipeline import Pipeline
from timeflux_ml.utils.import_helpers import make_object


def make_pipeline(steps, params):
    """

    Args:
        steps (dict):  (name, module_name, method_name) Tuples to specify steps of the pipeline to fit.
        params (dict): string -> object.  Parameters passed to the fit method of
                                          each step, where each parameter name is prefixed
                                          such that parameter `p` for step `s` has key `s__p`.
    Returns:
        pipeline: sklearn Pipeline object.

    """
    step_estimators = []
    for step_name, step_fullname in steps.items():
        step_estimator = make_object(step_fullname)
        step_estimators.append((step_name, step_estimator))
    pipeline = Pipeline(steps=step_estimators)
    try:
        pipeline.set_params(**params)
    except ValueError:
        raise ValueError("Could not set params of pipeline. Check the validity. ")
    return pipeline

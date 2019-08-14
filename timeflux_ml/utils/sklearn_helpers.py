from importlib import import_module
from sklearn.pipeline import Pipeline

def construct_pipeline(steps, params):
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
    for step_name, step_method in steps.items():
        module_name, estimator_name = step_method.rsplit('.', 1)

        try:
            module = import_module(module_name)
        except ImportError:
            raise ImportError(f'Could not import module {module_name}')
        try:
            step_estimator = getattr(module, estimator_name)()
        except AttributeError:
            raise ValueError(f'Module {module_name} has no object {estimator_name}')
        step_estimators.append((step_name, step_estimator))
    pipeline = Pipeline(steps=step_estimators)
    try:
        pipeline.set_params(**params)
    except ValueError:
        raise ValueError("Could not set params of pipeline. Check the validity. ")
    return pipeline
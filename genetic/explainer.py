import numpy as np
import shap

class Explainer:
    def __init__(self, model):
        self.model = model

        _model_class = str(type(model))
        if "xgboost.sklearn.XGBClassifier" in _model_class:
            self.explainer = lambda f, b: shap.TreeExplainer(model, data=b, model_output="probability")(f)
        if "xgboost.sklearn.XGBRegressor" in _model_class:
            self.explainer = lambda f, b: shap.TreeExplainer(model, data=b, model_output="raw")(f)


    def GSV(self, S_0, S_1):
        return self.explainer(S_0, S_1).values.mean(axis=0)


    def GSV_pop(self, S_0, S_1_pop):
        S_1_long = S_1_pop.reshape((S_1_pop.shape[0], S_1_pop.shape[1]*S_1_pop.shape[2]))
        return np.apply_along_axis(
            lambda S_1_long, S_0, d: self.GSV(S_0, S_1_long.reshape((d[0], d[1]))),
            1, S_1_long, S_0=S_0, d=S_1_pop[0].shape
        )


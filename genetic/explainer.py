import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost
class Explainer:
    def __init__(self, model):
        self.model = model

        model_class = type(model)
        if model_class in [xgboost.XGBClassifier, RandomForestClassifier]:
            self.explainer = lambda f, b: shap.TreeExplainer(model, data=b, model_output="probability")(f)
        if model_class in [xgboost.XGBRegressor, RandomForestRegressor]:
            self.explainer = lambda f, b: shap.TreeExplainer(model, data=b, model_output="raw")(f)


    def GSV(self, S_0, S_1):
        vals = self.explainer(S_0, S_1).values
        if vals.ndim == 2:
            return vals.mean(axis=0)
        else:
            return vals[..., 1].mean(axis=0)


    def GSV_pop(self, S_0, S_1_pop):
        S_1_long = S_1_pop.reshape((S_1_pop.shape[0], S_1_pop.shape[1]*S_1_pop.shape[2]))
        return np.apply_along_axis(
            lambda S_1_long, S_0, d: self.GSV(S_0, S_1_long.reshape((d[0], d[1]))),
            1, S_1_long, S_0=S_0, d=S_1_pop[0].shape
        )


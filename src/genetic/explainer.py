import numpy as np

from ..utils import tree_shap
class Explainer:
    def __init__(self, model):
        self.model = model


    def GSV(self, S_0, S_1, ordinal_encoder, ohe_encoder):
        LSV = tree_shap(self.model, S_0, S_1, ordinal_encoder, ohe_encoder)
        return LSV.mean(1).mean(1)


    def GSV_pop(self, S_0, S_1_pop, ordinal_encoder, ohe_encoder):
        N_pop, M, d = S_1_pop.shape
        S_1_aug = S_1_pop.reshape((M * N_pop, d))
        LSV = tree_shap(self.model, S_0, S_1_aug, ordinal_encoder, ohe_encoder)
        GSV = LSV.mean(1)
        GSV = GSV.reshape((d, N_pop, M)).mean(-1)
        return GSV.T
        # # S_1_long = S_1_pop.reshape((S_1_pop.shape[0], S_1_pop.shape[1]*S_1_pop.shape[2]))
        # return np.apply_along_axis(
        #     lambda S_1_long, S_0, d: self.GSV(S_0, S_1_long.reshape((d[0], d[1]))),
        #     1, S_1_long, S_0=S_0, d=S_1_pop[0].shape
        # )


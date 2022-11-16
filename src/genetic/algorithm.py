import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import xgboost

from .explainer import Explainer
from ..utils import audit_detection


class Algorithm:
    def __init__(
            self,
            model,
            S_0, S_1, 
            f_S_0, f_D_0, f_D_1,
            s_idx,
            constant=None,
            ordinal_encoder=None,
            ohe_encoder=None
        ):

        self.explainer = Explainer(model)
        self.S_0 = S_0
        self.S_1 = S_1
        self.f_S_0 = f_S_0
        self.f_D_0 = f_D_0
        self.f_D_1 = f_D_1
        self.s_idx = s_idx
        self.M, self.d = self.S_0.shape
        self.ordinal_encoder = ordinal_encoder
        self.ohe_encoder = ohe_encoder

        if constant is not None:
            self._idc = []
            if isinstance(constant[0], str):
                for const in constant:
                    self._idc.append(S_0.columns.get_loc(const))
            elif isinstance(constant[0], int):
                self._idc = constant
            else:
                raise Exception("constant must either contain indices of column names")
        else:
            self._idc = None

        self.result_explanation = {'original': None, 'changed': None}
        self.result_data = None
        self.iter_log = {'iter' : [], 'loss' : [], 'detection' : []}


    def fool(self, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        self.result_explanation['original'] = self.explainer.GSV(self.S_0, self.S_1, self.ordinal_encoder, self.ohe_encoder)
        self.result_explanation['changed'] = self.result_explanation['original']
        self.best_obj = np.abs(self.result_explanation['changed'][self.s_idx])


    # #:# plots 
    # def plot_data(self, i=0, constant=False, height=2, savefig=None):
    #     plt.rcParams["legend.handlelength"] = 0.1
    #     _colors = sns.color_palette("Set1").as_hex()[0:2][::-1]
    #     if i == 0:
    #         _df = self.result_data
    #     else:
    #         _data_changed = pd.DataFrame(self.get_best_data(i), columns=self.explainer.data.columns)
    #         _df = pd.concat((self.explainer.data, _data_changed))\
    #                 .reset_index(drop=True)\
    #                 .rename(index={'0': 'original', '1': 'changed'})\
    #                 .assign(dataset=pd.Series(['original', 'changed'])\
    #                                 .repeat(self._n).reset_index(drop=True))
    #     if not constant and self._idc is not None:
    #         _df = _df.drop(_df.columns[self._idc], axis=1)
    #     ax = sns.pairplot(_df, hue='dataset', height=height, palette=_colors)
    #     ax._legend.set_bbox_to_anchor((0.62, 0.64))
    #     if savefig:
    #         ax.savefig(savefig, bbox_inches='tight')
    #     plt.show()


    def plot_losses(self):
        # Curves of Shapley values
        fig, ax = plt.subplots()
        iters = self.iter_log['iter']
        ax.plot(iters, self.iter_log['loss'], 'r-')
        # set x-axis label
        ax.set_xlabel("Iterations")
        # set y-axis label
        ax.set_ylabel("Amplitude", color="red")
        ax.tick_params(axis='y', labelcolor="red")
        ax2 = ax.twinx()
        ax2.plot(iters, self.iter_log['detection'], 'b-')
        ax2.set_ylabel("Detection", color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")


    # New signature self.detector(S_1) -> {True, False}
    def detector(self, S_1):
        if isinstance(self.explainer.model, xgboost.XGBClassifier):
            if self.ohe_encoder is None:
                f_S_1 = self.explainer.model.predict(S_1, output_margin=True).reshape((-1, 1))
            else:
                f_S_1 = self.explainer.model.predict(self.ohe_encoder.transform(S_1), output_margin=True).reshape((-1, 1))
        else:
            if self.ohe_encoder is None:
                f_S_1 = self.explainer.model.predict_proba(S_1)[:, [1]]
            else:
                f_S_1 = self.explainer.model.predict_proba(self.ohe_encoder.transform(S_1))[:, [1]]
        
        return audit_detection(self.f_D_0, self.f_D_1, self.f_S_0, f_S_1, significance=0.05)
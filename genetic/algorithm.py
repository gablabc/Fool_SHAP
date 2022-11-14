import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Algorithm:
    def __init__(
            self,
            explainer,
            S_0, S_1, s_idx,
            detector,
            constant=None,
        ):

        self.explainer = explainer
        self.S_0 = S_0
        self.S_1 = S_1
        self.s_idx = s_idx
        self.detector = detector
        self.M, self.d = self.S_0.shape

        self.col_id = list(range(self.d))

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

        self.result_explanation['original'] = self.explainer.GSV(self.S_0, self.S_1)
        self.result_explanation['changed'] = np.zeros_like(self.result_explanation['original'])


    def fool_aim(self, random_state=None):
        Algorithm.fool(self=self, random_state=random_state)


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


    def plot_losses(self, lw=3, figsize=(9, 6)):
        plt.rcParams["figure.figsize"] = figsize
        plt.plot(
            self.iter_log['iter'], 
            self.iter_log['loss'],
            color='#000000',
            lw=lw
        )
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('Sensitive Feature Attribution', fontsize=16)
        plt.show()

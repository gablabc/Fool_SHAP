import numpy as np
import pandas as pd
import tqdm

from .algorithm import * 
from .utils import check_early_stopping


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        model,
        S_0, S_1, 
        f_S_0, f_D_0, f_D_1,
        s_idx,
        constant=None,
        ordinal_encoder=None,
        ohe_encoder=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            S_0=S_0, S_1=S_1, 
            f_S_0=f_S_0, f_D_0=f_D_0, f_D_1=f_D_1,
            s_idx=s_idx,
            constant=constant,
            ordinal_encoder=ordinal_encoder,
            ohe_encoder=ohe_encoder
        )

        # Setup
        params = dict(
            epsilon=1e-5,
            stop_iter=10,
            pop_count=50,
            std_ratio=1/9,
            mutation_prob=0.5,
            mutation_with_constraints=True,
            crossover_ratio=0.5,
            top_survivors=2,
            alpha=0.5
        )
        for k, v in kwargs.items():
            params[k] = v
        self.params = params
        
        # prepare std vector for mutation
        self._X_std = self.S_1.std(axis=0) * params['std_ratio']
        if self._idc is not None:
            for c in self._idc:
                self._X_std[c] = 0
        
        if params['mutation_with_constraints']:
            self._X_minmax = {
                'min': np.amin(self.S_1, axis=0), 
                'max': np.amax(self.S_1, axis=0)
            }
        
        # calculate probs for rank selection method
        self._rank_probs = np.arange(params['pop_count'], 0, -1) /\
             (params['pop_count'] * (params['pop_count'] + 1) / 2)
        self.fresh = True
        self.iter = 0
        

    #:# algorithm
    def fool(
        self,
        max_iter=50,
        random_state=None,
        verbose=True,
    ):
        
        # Init population
        if self.fresh:
            self.S_1_pop = np.tile(self.S_1, (self.params['pop_count'], 1, 1))
            self.E_pop = np.tile(self.result_explanation['original'], (self.S_1_pop.shape[0], 1)) 
            self.L_pop = np.zeros(self.params['pop_count']) 
            #self.mutation(adjust=3)
            self.log_losses()
            self.fresh = False
        
        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for _ in pbar:
            self.crossover()
            self.mutation()
            self.evaluation()
            if self.iter != max_iter:
                self.selection()

            self.log_losses()
            pbar.set_description("Iter: %s || Loss: %s" % (self.iter, self.iter_log['loss'][-1]))
            if check_early_stopping(self.iter_log, self.params['epsilon'], self.params['stop_iter']):
                break
            if self.iter >= 10 and np.sum(self.iter_log['detection'][-10:]) == 10:
                break

        # self.result_data = pd.concat((self.S_1, self.S_1_prime))\
        #     .reset_index(drop=True)\
        #     .rename(index={'0': 'original', '1': 'changed'})\
        #     .assign(dataset=pd.Series(['original', 'changed'])\
        #                     .repeat(self.M).reset_index(drop=True))

    def fool_aim(
            self,
            max_iter=50,
            random_state=None,
            verbose=True
        ):
        # First time the optimization is called
        if self.fresh:
            super().fool(
                random_state=random_state
            )
        self.fool(
            max_iter=max_iter, 
            random_state=random_state, 
            verbose=verbose, 
        )


    #:# inside
    
    def mutation(self, adjust=1):   
        _temp_pop_count = self.S_1_pop.shape[0]         
        _theta = np.random.normal(
            loc=0,
            scale=self._X_std * adjust,
            size=(_temp_pop_count, self.M, self.d)
        )
        # preserve zeros
        _theta = np.where(self.S_1_pop == 0, 0, _theta)
        # column mask made with the probability 
        _mask = np.random.binomial(
            n=1,
            p=self.params['mutation_prob'], 
            size=(_temp_pop_count, 1, self.d)
        )
        self.S_1_pop += _theta * _mask
        
        if self.params['mutation_with_constraints']:
            # Add min/max constraints for the variable distribution
            # this feature may lead to a much longer computation time
            S_1_pop_long = self.S_1_pop.reshape(_temp_pop_count * self.M, self.d)
            _X_long = np.tile(self.S_1, (_temp_pop_count, 1))
            for i in range(self.d):
                _max_mask = S_1_pop_long[:, i] > self._X_minmax['max'][i]
                _min_mask = S_1_pop_long[:, i] < self._X_minmax['min'][i]
                S_1_pop_long[:, i][_max_mask] = np.random.uniform(
                    _X_long[:, i][_max_mask],
                    np.repeat(self._X_minmax['max'][i], _max_mask.sum())
                )
                S_1_pop_long[:, i][_min_mask] = np.random.uniform(
                    np.repeat(self._X_minmax['min'][i], _min_mask.sum()),
                    _X_long[:, i][_min_mask]
                )
            self.S_1_pop = S_1_pop_long.reshape(_temp_pop_count, self.M, self.d)


    def crossover(self):
        # indexes of subset of columns (length is between 0 and p/2)
        _idv = np.random.choice(
            np.arange(self.d),
            size=np.random.choice(int(self.d / 2)),
            replace=False
        )
        # indexes of subset of population
        _idpop = np.random.choice(
            self.params['pop_count'], 
            size=int(self.params['pop_count'] * self.params['crossover_ratio']),
            replace=False
        )
        # get shuffled population
        _childs = self.S_1_pop[_idpop, :, :]
        # swap columns
        _childs[:, :, _idv] = _childs[::-1, :, _idv]
        self.S_1_pop = np.concatenate((self.S_1_pop, _childs))


    def evaluation(self):
        self.E_pop = self.explainer.GSV_pop(self.S_0, self.S_1_pop, self.ordinal_encoder, self.ohe_encoder)
        self.L_pop = np.abs(self.E_pop[:, self.s_idx])


    def selection(self):
        #:# take n best individuals and use p = i/(n*(n-1))
        _top_survivors = self.params['top_survivors']
        _top_f_ids = np.argpartition(self.L_pop, _top_survivors)[:_top_survivors]
        _random_ids = np.random.choice(
            self.params['pop_count'], 
            size=self.params['pop_count'] - _top_survivors, 
            replace=True,
            p=self._rank_probs
        )
        _sorted_ids = np.argsort(self.L_pop)[_random_ids]
        self.S_1_pop = np.concatenate((
            self.S_1_pop[_sorted_ids],
            self.S_1_pop[_top_f_ids]
        ))
        self.L_pop = np.concatenate((
            self.L_pop[_sorted_ids],
            self.L_pop[_top_f_ids]
        ))
        assert self.S_1_pop.shape[0] == self.params['pop_count'], 'wrong selection'
    

    #:# helper
    
    def get_best_idx(self):
        return np.argsort(self.L_pop)[0]


    def log_losses(self):
        # Log the results of the current generation
        best_idx = self.get_best_idx()
        curr_obj = np.abs(self.E_pop[best_idx, self.s_idx])
        self.iter_log['iter'].append(self.iter)
        self.iter_log['loss'].append(curr_obj)
        self.iter_log['detection'].append(self.detector(self.S_1_pop[best_idx]))
        self.iter += 1

        # Update the optimal undetected solution
        if not self.iter_log['detection'][-1] and curr_obj < self.best_obj:
            self.best_obj = curr_obj
            self.result_explanation['changed'] = self.E_pop[best_idx]
            self.S_1_prime = self.S_1_pop[best_idx]




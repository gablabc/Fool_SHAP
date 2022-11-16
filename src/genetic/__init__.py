""" 
Genetic Algorithm from the paper

H. Baniecki, P. Biecek. Manipulating SHAP via Adversarial Data Perturbations 
(Student Abstract). In: AAAI Conference on Artificial Intelligence (AAAI), 
36(11):12907-12908, 2022. URL: https://ojs.aaai.org/index.php/AAAI/article/view/21590.

Manipulate SHAP values by perturbing the background dataset via a genetic algorithm

Code cloned from https://github.com/hbaniecki/manipulating-shap
"""

from .explainer import Explainer
from .genetic import GeneticAlgorithm

all = ['Explainer', 'GeneticAlgorithm']
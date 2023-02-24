#!/bin/bash

python3 3_2_plot_all_results.py

# COMPAS
python3 3_3_plot_genetic_results.py --model=rf --dataset=compas --window_phi=10 --window_detect=20 --max_iter=200
python3 3_3_plot_genetic_results.py --model=xgb --dataset=compas --window_phi=20 --window_detect=30 --max_iter=400

# Adult
python3 3_3_plot_genetic_results.py --model=rf --dataset=adult_income --window_phi=10 --window_detect=20 --max_iter=100
python3 3_3_plot_genetic_results.py --model=xgb --dataset=adult_income --window_phi=20 --window_detect=30 --max_iter=400

# Marketing
python3 3_3_plot_genetic_results.py --model=rf --dataset=marketing --window_phi=2 --window_detect=2 --max_iter=12
python3 3_3_plot_genetic_results.py --model=xgb --dataset=marketing --window_phi=20 --window_detect=20 --max_iter=400

# Communities
python3 3_3_plot_genetic_results.py --model=rf --dataset=communities --window_phi=4 --window_detect=4 --max_iter=22
python3 3_3_plot_genetic_results.py --model=xgb --dataset=communities --window_phi=10 --window_detect=30 --max_iter=108

.PHONY : all
all : tree_shap fool_shap

tree_shap:
	python3 setup.py build
fool_shap:
	g++ src/fool_shap/main.cc -o src/fool_shap/main -std=c++11

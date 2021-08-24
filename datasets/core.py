import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter


import os
import random


def save(df, dataset, decision, rseed):

    df = shuffle(df, random_state=99)

    outdir = './preprocessed/{}/'.format(dataset)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #plain
    full_name    = '{}{}_full.csv'.format(outdir, dataset)

    train_name   = '{}{}_train_{}.csv'.format(outdir, dataset, rseed)
    test_name    = '{}{}_test_{}.csv'.format(outdir, dataset, rseed)
    explain_name  = '{}{}_explain_{}.csv'.format(outdir, dataset, rseed)
    

    #onehot
    oneHot_full_name    = '{}{}_fullOneHot.csv'.format(outdir, dataset)

    oneHot_train_name   = '{}{}_trainOneHot_{}.csv'.format(outdir, dataset, rseed)
    oneHot_test_name    = '{}{}_testOneHot_{}.csv'.format(outdir, dataset, rseed)
    oneHot_explain_name  = '{}{}_explainOneHot_{}.csv'.format(outdir, dataset, rseed)

    

    # one_hot_df
    df_onehot = pd.get_dummies(df)


    # plain data
    df_train, df_holdout, indices_train, indices_holdout = train_test_split(df, range(len(df)), test_size=0.33, random_state=rseed, stratify=df[decision])
    df_test, df_explain, indices_test, indices_explain = train_test_split(df_holdout, range(len(df_holdout)), test_size=0.5, random_state=rseed, stratify=df_holdout[decision])

    # onehot data
    df_onehot_train = df_onehot.iloc[indices_train,:]
    df_onehot_holdout  = df_onehot.iloc[indices_holdout,:]
    df_onehot_test, df_onehot_explain = df_onehot_holdout.iloc[indices_test,:], df_onehot_holdout.iloc[indices_explain,:]

    
    #save the full dataset
    df.to_csv(full_name, index=False)
    df_onehot.to_csv(oneHot_full_name, index=False)
    


    #save train set 
    df_train.to_csv(train_name, index=False)
    df_onehot_train.to_csv(oneHot_train_name, index=False)
    


    #save test set
    df_test.to_csv(test_name, index=False)
    df_onehot_test.to_csv(oneHot_test_name, index=False)
    

    #save explain set
    df_explain.to_csv(explain_name, index=False)
    df_onehot_explain.to_csv(oneHot_explain_name, index=False)
    

    
    
    
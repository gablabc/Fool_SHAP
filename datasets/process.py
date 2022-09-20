import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_adult_income():

    # output files
    dataset = 'adult_income'

    raw_data_1 = np.genfromtxt('./raw_datasets/adult_income/adult.data', delimiter=', ', dtype=str)
    raw_data_2 = np.genfromtxt('./raw_datasets/adult_income/adult.test', delimiter=', ', dtype=str, skip_header=1)

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    df_1 = pd.DataFrame(raw_data_1, columns=column_names)
    df_2 = pd.DataFrame(raw_data_2, columns=column_names)
    df = pd.concat([df_1, df_2], axis=0)


    # For more details on how the below transformations 
    df = df.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64, "capital-gain": np.int64, "capital-loss": np.int64 })

    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    df = df.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    df = df.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    df = df.replace({'workclass': {'?': 'Other/Unknown'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                           'Exec-managerial':'White-Collar','Farming-fishing':'Blue-Collar',
                                            'Handlers-cleaners':'Blue-Collar',
                                            'Machine-op-inspct':'Blue-Collar','Other-service':'Service',
                                            'Priv-house-serv':'Service',
                                           'Prof-specialty':'Professional','Protective-serv':'Service',
                                            'Tech-support':'Service',
                                           'Transport-moving':'Blue-Collar','Unknown':'Other/Unknown',
                                            'Armed-Forces':'Other/Unknown','?':'Other/Unknown'}})

    df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married','Never-married':'Single'}})


    df = df[['age','workclass','education', 'marital-status', 'relationship', 'occupation', 'race', 'gender', 'capital-gain', 'capital-loss',  'hours-per-week', 'income']]

    #df = df[['age','workclass','education', 'marital-status', 'relationship', 'occupation', 'race', 'gender',  'hours-per-week', 'income']]

    df = df.replace({'income': {'<=50K': 0, '<=50K.': 0,  '>50K': 1, '>50K.': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                           '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                          '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})


    df = df.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week', 'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss'})
    #df = df.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    save(df, dataset)



def get_marketing():
    # output files
    dataset = 'marketing'
    decision = 'subscribed'

    df = pd.read_csv('./raw_datasets/marketing/marketing.csv')
    
    df['age'] = df['age'].apply(lambda x: 'age:30-60' if ((x >= 30) & (x <=60))  else 'age:not30-60')
    

    save(df, dataset)



def get_default_credit():
    # output files
    dataset = 'default_credit'
    decision = 'good_credit'

    df = pd.read_csv('./raw_datasets/default_credit/default_credit.csv')

    
    df['good_credit'] = 1 - df['DEFAULT_PAYEMENT']

    df = df.drop(labels=['DEFAULT_PAYEMENT'], axis = 1)

    df = df.dropna()

    #missing_fractions = df.isnull().mean().sort_values(ascending=False) # Fraction of data missing for each variable
    #print(missing_fractions[missing_fractions > 0]) # Print variables that are missing data


    save(df, dataset)




def get_compas(save_df=False):

    # output files
    dataset = 'compas'

    """Loads COMPAS dataset from https://raw.githubusercontent.com/algofairness/fairness-comparison/master/fairness/data/raw/propublica-recidivism.csv
    :param: save_intermediate: save the transformed dataset. Do not save by default.
    """
    url = "https://raw.githubusercontent.com/algofairness/fairness-comparison/master/fairness/data/raw/propublica-recidivism.csv"

    df = pd.read_csv(url)

    df['low_risk'] = 1 - df['two_year_recid']

    df = df[['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'low_risk']]

    df.to_csv('./raw_datasets/compas/compas.csv', index=False)

    print(len(df))

    df = df[(df['race']=='African-American') | (df['race']=='Caucasian')]

    print(len(df))

    df = df.replace({'c_charge_degree': {'M': 'Misdemeanor', 'F': 'Felony'}})

    save(df, dataset)



def get_communities():
    """"
    Taken from https://github.com/dylan-slack/Fooling-LIME-SHAP
    Handle processing of Communities and Crime.  We exclude rows with missing values and predict
    if the violent crime is in the 50th percentile.

    Parameters
    ----------
    params : Params

    Returns:
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """

    dataset = 'communities'
    df = pd.read_csv('./raw_datasets/communities/c&c.csv', index_col=0)
    # Rename the columns
    df.columns = [x.split(" ")[0] for x in df.columns]

    # Considre racePctWhite > 95 as sensitive attribute
    df.insert(loc=0, column='PctWhite>90', value=(df['racePctWhite'] > 90).astype(int))
    df = df.drop(['racepctblack', 'racePctAsian', 'racePctWhite', 'racePctAsian'], axis=1)

    # Remove missing targets
    y_col = "ViolentCrimesPerPop"
    df = df[df[y_col] != "?"]
    df[y_col] = df[y_col].values.astype('float32')

    # Just dump all x's that have missing values 
    cols_with_missing_values = []
    for col in df:
        if len(np.where(df[col].values == '?')[0]) >= 1:
            cols_with_missing_values.append(col)    
    df = df.drop(cols_with_missing_values + ['communityname', 'fold', 'county', 'community', 'state'], axis=1)


    # Everything over 50th percentile gets negative outcome (lots of crime is bad)
    high_violent_crimes_threshold = 50
    y_col = "ViolentCrimesPerPop"
    y = df[y_col]
    y_cutoff = np.percentile(y, high_violent_crimes_threshold)
    df[y_col] = [1 if val < y_cutoff else 0 for val in df[y_col]]
    df.rename({"ViolentCrimesPerPop": 'LowCrime'}, axis=1, inplace=True)
    
    save(df, dataset)



def save(df, dataset):
    df = shuffle(df, random_state=99)
    full_name = os.path.join("preprocessed", f"{dataset}.csv")
    # Save the whole processed dataset
    df.to_csv(full_name, index=False)
    # Save the idxs of train/test for 5 splits
    for s in range(5):
        train_idx, test_idx = train_test_split(list(range(len(df))), test_size=0.2,
                                                            random_state=s, stratify=df.iloc[:, -1])
        json.dump({"train" : train_idx, "test" : test_idx},
                   open(os.path.join("preprocessed", f"{dataset}_split_rseed_{s}.json"), "w"))



DATA_PROCESS = {
    'adult_income' : get_adult_income,
    'default_credit' : get_default_credit,
    'marketing' : get_marketing,
    'compas' : get_compas,
    'communities' : get_communities
}
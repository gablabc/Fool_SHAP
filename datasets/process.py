import numpy as np
import pandas as pd
from sklearn.utils import shuffle
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



def save(df, dataset):
    df = shuffle(df, random_state=99)
    full_name = os.path.join("preprocessed", f"{dataset}.csv")
    df.to_csv(full_name, index=False)



DATA_PROCESS = {
    'adult_income' : get_adult_income,
    'default_credit' : get_default_credit,
    'marketing' : get_marketing,
    'compas' : get_compas
}
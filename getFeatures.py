
import numpy as np
import pandas as pd
from datetime import datetime
import math

def yaniv_to_ed_query(query):

    query[0]['people'] = query[0].pop('patients')

    for patient in query[0]['people']:
        patient['attributes']['people.dateOfBirth'] = patient['attributes'].pop('birth')
        patient['attributes']['people.dateOfDeath'] = patient['attributes'].pop('death')
        patient['attributes']['people.@gender'] = [(
            'F' if patient['attributes'].pop('gender') == 'female'
            else 'M')]
        patient['attributes']['people.@diagData'] = patient['attributes'].pop('@conditions')
        for condition in patient['attributes']['people.@diagData']:
            condition['diagnosisDate'] = condition.pop('date')
            condition['diagnosis'] = condition.pop('description')
        patient['attributes'].pop('name')
        patient['attributes'].pop('patient_id')
        
    return query


def get_conditions(query, startDate='1900-01-01', endDate='2019-12-31'):
        
    startDate = datetime.strptime(startDate, '%Y-%m-%d')
    endDate = datetime.strptime(endDate, '%Y-%m-%d')

    conditions = [
        condition['diagnosis']
        for patient in query[0]['people'] 
        for condition in patient['attributes']['people.@diagData']
        if (
            datetime.strptime(condition['diagnosisDate'], '%Y-%m-%d %H:%M:%S') 
            >= startDate
            and datetime.strptime(condition['diagnosisDate'], '%Y-%m-%d %H:%M:%S')
            <= endDate
        )
    ]

    conditions = pd.Series(conditions).value_counts()    

    return conditions


def get_live_patients(query, startDate='1900-01-01', endDate='2019-12-31'):

    startDate = datetime.strptime(startDate, '%Y-%m-%d')
    endDate = datetime.strptime(endDate, '%Y-%m-%d')

    patients = [
        patient['v_id'] 
        for patient in query[0]['people']
        if (
            datetime.strptime(
                patient['attributes']['people.dateOfBirth'], '%Y-%m-%d %H:%M:%S') 
            <= endDate
            and datetime.strptime(
                patient['attributes']['people.dateOfDeath'], '%Y-%m-%d %H:%M:%S')
            >= startDate
        )
    ]

    return patients

def make_age_groups(years = 5, top_year = 100):
    age_groups_ranges = [(i,min(i+years-1,top_year)) 
        for i in range(0, top_year, years)] + [(top_year, 140)]

    age_group_titles = [
        'Age {}-{}'.format(start, end) for start, end in age_groups_ranges]

    return age_groups_ranges, age_group_titles


def get_feature_vec(query, conditions, startDate, endDate, age_groups):

    startDate = datetime.strptime(startDate, '%Y-%m-%d')
    endDate = datetime.strptime(endDate, '%Y-%m-%d')

    demog_df = pd.DataFrame([patient['attributes'] 
                             for patient in query[0]['people']])

    demog_df.index = [
        patient['v_id'] for patient in query[0]['people']
    ]

    demog_df = demog_df[[
        'people.@gender',
        'people.dateOfBirth',
        'people.dateOfDeath',
    ]]

    df = pd.DataFrame(
            np.zeros((len(demog_df.index), len(conditions.index))),
            index=demog_df.index, 
            columns=conditions.index,
        )

    for patient in query[0]['people']:

        patient_conditions = [
            condition['diagnosis'] 
            for condition in patient['attributes']['people.@diagData']
            if (
                datetime.strptime(condition['diagnosisDate'], '%Y-%m-%d %H:%M:%S') 
                >= startDate
                and datetime.strptime(condition['diagnosisDate'], '%Y-%m-%d %H:%M:%S')
                <= endDate
            )
        ]

        df.loc[patient['v_id'], patient_conditions ] = 1

    return concat_features(
        conditions_df=df, 
        demog_df=demog_df, 
        date=endDate, 
        age_groups=age_groups,
    )

def concat_features(conditions_df, demog_df, date, age_groups):

    dead_df = deceased(demog_df, date)
    gender_df = gender(demog_df)
    age_df = age_group_df(
        df = demog_df, 
        date_for_age = date,
        age_groups=age_groups,
    )
    
    return pd.concat([gender_df, dead_df, age_df, conditions_df], axis=1)


def age_group_df(df, date_for_age, age_groups):

    age_group_df = pd.DataFrame(
        np.zeros((len(df),len(age_groups[0]))),
        index=df.index, 
        columns=age_groups[1]
        )

    for i in df.index:
        age = math.floor(
            (date_for_age
            - datetime.strptime(df.loc[i,'people.dateOfBirth'], '%Y-%m-%d %H:%M:%S')
            ).days/365.25
        )

        for j, age_group in enumerate(age_groups[0]):
            if age >= age_group[0] and age <= age_group[1]:
                age_group_df.loc[i].iloc[j]=1

    return age_group_df


def deceased(df, date):

    dead = df['people.dateOfDeath'].apply(
        lambda x: 1.0 if (
            date - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        ).days > 0 else 0.0
    )

    dead.name = 'Deceased'

    return dead


def gender(df):

    gender = df['people.@gender'].apply(
        lambda x: 1.0 if x[0]=='F' else 0.0
    )

    gender.name = 'Female'

    return gender

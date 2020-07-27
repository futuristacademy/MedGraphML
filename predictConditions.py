import getFeatures
import feature_weighted_mse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import t    
from statsmodels.stats.multitest import multipletests

def predictConditions(query):
    
    print('Collecting all conditions:\n')
    
    conditions = getFeatures.get_conditions(
        query=query, startDate='2019-01-01', endDate='2020-12-31')
    
    print(conditions)
    

    patients = getFeatures.get_live_patients(
        query=query, startDate='2019-12-31', endDate='2019-12-31')
    
    print('\nNumber of patients', len(patients))
    

    age_groups = getFeatures.make_age_groups()
    
    print('\nAge groups\n', age_groups)
    

    print('\nCompute features: ')
    
    x_df = getFeatures.get_feature_vec(
        query,
        conditions=conditions,
        startDate='2019-01-01', 
        endDate='2019-12-31', 
        age_groups=age_groups)

    print('\nx_df.shape ', x_df.shape)
    
    print('\nCompute labels: ')
    
    y_df = getFeatures.get_feature_vec(
        query,
        conditions=conditions,
        startDate='2020-01-01', 
        endDate='2020-12-31', 
        age_groups=age_groups)
    
    print('\ny_df.shape ', y_df.shape)
    

    train, test = train_test_split(patients, test_size=0.25, random_state=42)
    
    x_train_df = x_df.loc[train]
    y_train_df = y_df.loc[train]
    x_test_df = x_df.loc[test]
    y_test_df = y_df.loc[test]
    
    print('\n\nTrain set:', len(train), 'Test set: ', len(test))
    
    print(
        '\n\nSorted x_train means:\n\n',
        x_train_df.mean().sort_values(ascending=False), 
        '\n\nSorted y_train means:\n\n',
        y_train_df.mean().sort_values(ascending=False)
    )

    filter_below = 20
    print('\nFiltereing conditions with less than {} cases:'.format(filter_below))
    
    x_drop_list = ( 
        set(x_train_df.columns[x_train_df.sum() < filter_below])
        | set(x_test_df.columns[x_train_df.sum() < filter_below])
    )

    x_train_df = x_train_df.drop(x_drop_list, axis=1)
    x_test_df = x_test_df.drop(x_drop_list, axis=1)

    y_drop_list = ( 
        set(y_train_df.columns[y_train_df.sum() < filter_below])
        | set(y_test_df.columns[y_train_df.sum() < filter_below])
    )

    y_train_df = y_train_df.drop(y_drop_list, axis=1)
    y_test_df = y_test_df.drop(y_drop_list, axis=1)

    print(
        '\n\nSorted x_train means:\n\n',
        x_train_df.mean().sort_values(ascending=False), 
        '\n\nSorted y_train means:\n\n\n\n',
        y_train_df.mean().sort_values(ascending=False)
    )
    
    print(
        '\n\nSorted x_test means:\n\n',
        x_test_df.mean().sort_values(ascending=False), 
        '\n\nSorted y_test means:\n\n\n\n',
        y_test_df.mean().sort_values(ascending=False)
    )

    y_weights = 1 / (y_train_df.var() + 1e-3)
    y_weights = y_weights/(y_train_df.var()*y_weights).sum()
    
    print(
        '\n',
        pd.DataFrame(
            [y_train_df.var(), y_weights, y_weights*y_train_df.var()],
             index=['y_train var', 'y_weights', 'var*weight']
        ).transpose()
    )

    wmse = feature_weighted_mse.make_feature_weighted_mse(y_weights)
    
    print(
        '\nBasic benchmark - y means\n', 
        'Train loss',
        wmse(
            y_true=y_train_df.values, 
            y_pred=y_train_df.values.mean(axis=0)
        ).numpy().mean(),
    )

    from sklearn.model_selection import RepeatedKFold

    n_splits = 4
    n_repeats = 2
    alpha=0.00001
    learning_rate=0.001
    patience=30
    
    print('\nTrain linear model using Lasso alpha {} {}-fold CV repeated {} times.\n'.format(
        alpha, n_splits, n_repeats,
    ))
    

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    models=[]
    history=[]
    performance=[]
    
    i=0
    for train_index, validate_index in rkf.split(x_train_df):
        
        i += 1
        print('\n\nFold {} out of {}\n\n'.format(i, n_splits*n_repeats))
        
        x_train, x_validate = x_train_df.iloc[train_index], x_train_df.iloc[validate_index]
        y_train, y_validate = y_train_df.iloc[train_index], y_train_df.iloc[validate_index]
    
        inputs = keras.layers.Input(shape=x_train_df.shape[1])
        outputs = keras.layers.Dense(
            units=y_train_df.shape[1], 
            kernel_regularizer=keras.regularizers.l1(l=alpha),
        )(inputs)
        
        models.append(keras.Model(inputs=inputs, outputs=outputs))

        models[-1].compile(loss=wmse, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        history.append(models[-1].fit(
            x=x_train,
            y=y_train,
            batch_size=128,
            epochs=1000,
            validation_data=(x_validate, y_validate),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ]
        ))
    
        print('\nEvaluate on test set:\n')
        performance.append(models[-1].evaluate(x=x_test_df, y=y_test_df))
        print(performance[-1],'\n')

    print('Test loss mean', np.mean(performance), 'std' , np.std(performance, ddof=1))
    

    
    constant_full = pd.DataFrame(
        np.array([model.layers[1].get_weights()[1] for model in models]).transpose(), 
        index=y_train_df.columns, 
        columns=['Fold {}'.format(i) for i in range(1, 1+n_splits*n_repeats)],
    )
    constant_full.to_csv('constant_full.csv')
    
    coef_mat = np.array([model.layers[1].get_weights()[0] for model in models]).transpose((1, 2, 0))
    
    coef_full = pd.DataFrame(
        [[json.dumps(coef_mat[i,j].tolist()) 
          for j in range(coef_mat.shape[1])] 
         for i in range(coef_mat.shape[0])], 
        columns=y_train_df.columns, 
        index=x_train_df.columns
    ).transpose()
    
    coef_full.to_csv('coef_full.csv')

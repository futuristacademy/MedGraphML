import getFeatures
import feature_weighted_mse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns

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
    train, validate = train_test_split(train, test_size=1/3, random_state=42)
    
    x_train_df = x_df.loc[train]
    y_train_df = y_df.loc[train]
    x_validate_df = x_df.loc[validate]
    y_validate_df = y_df.loc[validate]
    x_test_df = x_df.loc[test]
    y_test_df = y_df.loc[test]
    
    print('\n\nTrain set:', len(train), 'Validate set: ', len(validate), 'Test set: ', len(test))
    
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
        | set(x_validate_df.columns[x_train_df.sum() < filter_below])
    )

    x_train_df = x_train_df.drop(x_drop_list, axis=1)
    x_validate_df = x_validate_df.drop(x_drop_list, axis=1)
    x_test_df = x_test_df.drop(x_drop_list, axis=1)

    y_drop_list = ( 
        set(y_train_df.columns[y_train_df.sum() < filter_below])
        | set(y_test_df.columns[y_train_df.sum() < filter_below])
        | set(y_validate_df.columns[y_train_df.sum() < filter_below])
    )

    y_train_df = y_train_df.drop(y_drop_list, axis=1)
    y_validate_df = y_validate_df.drop(y_drop_list, axis=1)
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

    y_weights = 1 / (y_train_df.var() + 10e-7) / y_train_df.shape[-1]
    
    print(
        '\n',
        pd.DataFrame(
            [y_train_df.var(), y_weights],
             index=['y_train var', 'y_weights']
        ).transpose()
    )
    
    

    wmse = feature_weighted_mse.make_feature_weighted_mse(y_weights)
    
    print(
        '\nBasic benchmark #1 - y means\n', 
        'Train loss',
        wmse(
            y_true=y_train_df.values, 
            y_pred=y_train_df.values.mean(axis=0)
        ).numpy().mean(),
    )

    print('\nTrain linear model (Lasso)\n')
    
    inputs = keras.layers.Input(shape=x_train_df.shape[1])
    outputs = keras.layers.Dense(
        units=y_train_df.shape[1], 
        kernel_regularizer=keras.regularizers.l1(l=0.0002),
    )(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=wmse, optimizer=keras.optimizers.Adam())

    history = model.fit(
        x=x_train_df,
        y=y_train_df,
        batch_size=128,
        epochs=1000,
        validation_data=(x_validate_df, y_validate_df),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ]
    )
    
    print('\nEvaluate on test set:\n')
    print(model.evaluate(x=x_test_df, y=y_test_df))

    f, ax = plt.subplots(figsize=(50, 50))
    ax = sns.heatmap(
        model.layers[1].get_weights()[0].transpose(), 
        xticklabels=x_train_df.columns, 
        yticklabels=y_train_df.columns,
        center=0.0,
        cmap='seismic',
    )
    
    plt.savefig('linear_coefs.png')
    

    print('\nTrain non-linear model (1 hidden layer):\n')
    
    inputs = keras.layers.Input(shape=x_train_df.shape[1])
    x = keras.layers.Dense(
        units=128, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l1(l=0.0002),
    )(inputs)
    outputs = keras.layers.Dense(
        units=y_train_df.shape[1],
        kernel_regularizer=keras.regularizers.l1(l=0.0002),
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=wmse, optimizer=keras.optimizers.Adam())

    history = model.fit(
        x=x_train_df,
        y=y_train_df,
        batch_size=128,
        epochs=1000,
        validation_data=(x_validate_df, y_validate_df),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ]
    )
    
    print('\nEvaluate on test set:\n')
    print(model.evaluate(x=x_test_df, y=y_test_df))

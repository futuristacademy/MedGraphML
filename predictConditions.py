import getFeatures
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
    
    print(conditions[:20])
    
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
    
    print('Train set:', len(train), 'Validate set: ', len(validate), 'Test set: ', len(test))
    
    print(
        '\n',
        pd.DataFrame(
            [x_train_df.mean(), y_train_df.mean(), 
             x_validate_df.mean(), y_validate_df.mean()],
             index=['x_train means','y_train means',
                    'x_validate means','y_validate means']
        ).transpose().head(60)
    )


    print(
        '\nBasic benchmark - all conditions persist\n', 
        'Train loss',
        keras.losses.mse(y_train_df.values, x_train_df.values).numpy().mean(),
        'Validate loss',
        keras.losses.mse(y_validate_df.values, x_validate_df.values).numpy().mean()
    )
    
    
    print('\nTrain linear model (Lasso)\n')
    
    inputs = keras.layers.Input(shape=x_train_df.shape[1])
    outputs = keras.layers.Dense(
        units=y_train_df.shape[1], 
        kernel_regularizer=keras.regularizers.l1(l=0.0000001),
    )(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer=keras.optimizers.Adam())

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
        yticklabels=x_train_df.columns)
    
    plt.savefig('linear_coefs.png')
    
    print('\nTrain non-linear model (1 hidden layer):\n')
    
    inputs = keras.layers.Input(shape=x_train_df.shape[1])
    x = keras.layers.Dense(units=128, activation='relu')(inputs)
    outputs = keras.layers.Dense(units=y_train_df.shape[1])(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer=keras.optimizers.Adam())

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

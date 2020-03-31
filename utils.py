# -- coding: utf-8 --

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_descriptive_stat(df, var_list):
    """
    Creates descriptive statistics on a pandas DataFrame.

    :param df: pandas DataFrame
    :param var_list: list of variables to include
    """
    df = df[var_list]
    desc_df = df.describe(percentiles=np.arange(0.1, 1, 0.1).round(2))
    desc_df = desc_df.append(df.agg({np.median
                                        , lambda x: x.mode()[0]})).rename(index={'<lambda>': 'mode'}).transpose()
    desc_df['missing nbr'] = df.isnull().sum()
    desc_df['missing %'] = ((df.isnull().sum()) / df.shape[0] * 100)
    desc_df = desc_df[['min', 'max', 'mean', 'mode', 'median', '10%', '20%',
                       '30%', '40%', '50%', '60%', '70%', '80%', '90%', 'missing nbr', 'missing %']].transpose()

    return desc_df.round(2)

def plot_full_stacked_density(df, target_var, var, kind):
    """
    Plots a full stacked density plot of the given variable.

    :param df: pandas DataFrame
    :param target_var: target variable
    :param var: variable
    :param kind: the kind of the plot, e.g. 'bar', 'area', 'density'
    """
    pivot_df = df.groupby([target_var, var]).size().reset_index().pivot(columns=target_var, index=var, values=0)
    pivot_df = pivot_df.div(pivot_df.sum(1), axis=0)
    pivot_df.plot(kind=kind, stacked=True, rot=45, title='Density of churn')
    plt.ylabel('Density of churn (%)')

    return plt.show()

def calc_pearson_correlation(df, var_list):
    """
    Calculates Pearson correlation between variables. 
    
    :param df: pandas DataFrame
    :param var_list: list of variables
    """
    pearson_corr = df[var_list].apply(lambda x: pd.factorize(x)[0]).corr(method='pearson').abs() 
    return pearson_corr
    

def select_by_correlation_threshold(df, threshold):
    """
    List all variables with higher or equal to the correlation as the threshold.

    :param df: pandas DataFrame with the correlatting variable pairs
    :param threshold: threshold value
    """
    corr_sorted = df.unstack().sort_values(ascending=False).drop_duplicates()
    corr_df = pd.DataFrame(corr_sorted)
    corr_df = corr_df.reset_index().rename(columns={'level_0': 'variable_1', 
                                                    'level_1': 'variable_2', 
                                                    0: 'correlation'})
    corr_df = corr_df[corr_df['correlation'] != 1]
    corr_df = corr_df[corr_df.correlation >= threshold]
    return corr_df


def variable_selection(selected, target, train, nfolds, n, model_type):
    """
    Performs variable selection based on the selected model type by the user.  
    Returns a variable importance plot and a list with the n most important variables. 

    :param selected: list of selected variables
    :param target: target variable
    :param train: h2o frame with train data
    :param nfolds: the number of folds to use for cross-validation
    :param n: number of important variables to be selected at the end of the training process
    :param model_type: the name of the model the variable selection should be calculated on (can be 'GBM', 'GLM', 'DRF')
    """
    if model_type == 'GBM':
        model = H2OGradientBoostingEstimator(balance_classes=True,
                                             seed=1234,
                                             model_id='GBM_var_sel',
                                             nfolds=nfolds,
                                             stopping_rounds=3,
                                             stopping_tolerance=0.01,
                                             stopping_metric="lift_top_group")

        model.train(x=selected, y=target, training_frame=train)

        varimp = model.varimp(use_pandas=True)[0:n + 1]
        result = list(varimp['variable'])

    if model_type == 'GLM':
        model = H2OGeneralizedLinearEstimator(family='binomial',
                                              alpha=1.0,  # lasso regularization
                                              lambda_search=True,
                                              nfolds=nfolds,
                                              standardize=True,
                                              seed=1234,
                                              model_id='GLM_var_sel')

        model.train(x=selected, y=target, training_frame=train)

        varimp = model.coef_norm()
        glm_df = pd.DataFrame(varimp.items(), columns=['variable', 'scaled_importance'])
        glm_df = glm_df[glm_df.scaled_importance > 0.0]
        glm_df = glm_df.sort_values(by=['scaled_importance'], ascending=False)
        glm_df = glm_df[glm_df.variable != 'Intercept']
        result = glm_df['variable'][0:n]
        for i in range(len(result)):
            result.iloc[i] = result.iloc[i].split('.')[0]
        result = list(set(result))

    if model_type == 'DRF':
        model = H2ORandomForestEstimator(balance_classes=True,
                                         seed=1234,
                                         ntrees=10,
                                         model_id="DRF_var_sel",
                                         nfolds=nfolds,
                                         stopping_tolerance=0.01,
                                         stopping_metric="lift_top_group")

        model.train(x=selected, y=target, training_frame=train)

        varimp = model.varimp(use_pandas=True)[0:n]
        result = list(varimp['variable'])

    return model, result


def modelling_for_testing_variables(target, var_list, train, nfolds):
    """
    Trains 3 models (GBM, GLM, Random Forest) for testing a list of variables.

    :param selected: list of selected variables
    :param target: target variable
    :param train: h2o frame with train data
    :param nfolds: the number of folds to use for cross-validation

    """
    model_gbm = H2OGradientBoostingEstimator(balance_classes=True,
                                             seed=1234,
                                             nfolds=nfolds,
                                             stopping_tolerance=0.01,
                                             stopping_metric="lift_top_group")

    model_glm = H2OGeneralizedLinearEstimator(family='binomial',
                                              seed=1234,
                                              nfolds=nfolds,
                                              alpha=0.7,
                                              lambda_search=True)

    model_drf = H2ORandomForestEstimator(balance_classes=True,
                                         seed=1234,
                                         nfolds=nfolds,
                                         stopping_tolerance=0.01,
                                         stopping_metric="lift_top_group",
                                         ntrees=15)

    model_gbm.train(x=var_list, y=target, training_frame=train)
    model_glm.train(x=var_list, y=target, training_frame=train)
    model_drf.train(x=var_list, y=target, training_frame=train)

    model_gbm.name = 'GBM'
    model_glm.name = 'GLM'
    model_drf.name = 'DRF'

    return model_gbm, model_glm, model_drf


def result_comparison(model_list, test, target):
    """
    Compare results of different models on the test set.

    :param model_list: list of trained h2o models
    :param test: h2o frame with test data
    :param target: target variable
    """
    model_name = []
    cum_resp_rate_val = []
    cum_cap_rate_val = []
    cum_resp_rate_val_20 = []
    cum_cap_rate_val_20 = []
    auc_vars = []
    nbr = []
    target_nbr = []

    for i in model_list:
        auc = i.model_performance(test_data=test).auc()

        lift_df = i.model_performance(test_data=test).gains_lift().as_data_frame().iloc[5:6, :]
        lift_df_20 = i.model_performance(test_data=test).gains_lift().as_data_frame().iloc[7:8, :]

        cum_resp_rate_val.append(lift_df.iloc[0]['cumulative_response_rate'])
        cum_cap_rate_val.append(lift_df.iloc[0]['cumulative_capture_rate'])

        cum_resp_rate_val_20.append(lift_df_20.iloc[0]['cumulative_response_rate'])
        cum_cap_rate_val_20.append(lift_df_20.iloc[0]['cumulative_capture_rate'])

        model_name.append(i.name)
        nbr.append(test.shape[0])
        target_nbr.append(test[target].sum())
        auc_vars.append(auc)

    res_df = pd.DataFrame({'Model': model_name, 'Customer Nbr': nbr, 'Target Nbr': target_nbr, 'AUC': auc_vars,
                           'Cum Response Rate TOP 10%': cum_resp_rate_val,'Cum Response Rate TOP 20%': cum_resp_rate_val_20,
                           'Cum Capture Rate TOP 10%': cum_cap_rate_val, 'Cum Capture Rate TOP 20%': cum_cap_rate_val_20
                           })

    res_df = res_df.round(2)
    res_df['Target Nbr'] = res_df['Target Nbr'].astype(int)
    res_df['Target Found TOP 10%'] = (res_df['Target Nbr'] * res_df['Cum Capture Rate TOP 10%']).astype(int)
    res_df['Target Found TOP 20%'] = (res_df['Target Nbr'] * res_df['Cum Capture Rate TOP 20%']).astype(int)

    fin_df = res_df[['Model', 'AUC', 'Customer Nbr', 'Target Nbr',
                     'Cum Capture Rate TOP 10%', 'Cum Response Rate TOP 10%', 'Target Found TOP 10%',
                     'Cum Capture Rate TOP 20%', 'Cum Response Rate TOP 20%', 'Target Found TOP 20%']]

    return fin_df


def gbm_grid_search_max_depth(selected, target, train, test, nfolds):
    """
    Performs grid search on a GBM model to find the optimal max_depth parameter that maximizes the AUC on the test frame.
    Returns the best model and the grid. 

    :param selected: list of selected variables
    :param target: target variable
    :param train: h2o frame with train data
    :param test: h2o frame with test data
    :param nfolds: the number of folds to use for cross-validation
    """
    hyperparameters = {'max_depth': [3, 4, 5, 6, 7, 8]}
    search_criteria = {'strategy': "Cartesian"}
    gbm_grid = H2OGridSearch(H2OGradientBoostingEstimator(seed=1234,
                                                          balance_classes=True,
                                                          nfolds=nfolds),
                             hyperparameters,
                             search_criteria=search_criteria)

    gbm_grid.train(x=selected, y=target, training_frame=train, validation_frame=test)

    gbm_grid_table = gbm_grid.get_grid(sort_by='auc', decreasing=True)
    gbm_best_model = gbm_grid.models[0]
    gbm_best_model.name = 'best GBM - max_detph'
    gbm_grid_table = gbm_grid_table.sorted_metric_table().drop('model_ids', axis=1)

    return gbm_best_model, gbm_grid_table


def gbm_grid_search(selected, target, train, test, nfolds, max_depth, ntrees_list, min_rows_list, show_top):
    """
    Performs grid search on a GBM model to find the optimal parameters that maximize the AUC on the test frame.
    Returns the best model and the grid. 

    :param selected: list of selected variables
    :param target: target variable
    :param train: h2o frame with train data
    :param test: h2o frame with test data
    :param nfolds: the number of folds to use for cross-validation
    :param max_depth: specifies the maximum depth to which each tree will be built
    :param ntrees_list: list of number of trees to build in the model
    :param min_rows_list: list of the minimum number of observations for a leaf in order to split
    """
    hyperparameters = {'ntrees': ntrees_list,
                       'min_rows': min_rows_list,
                       'min_split_improvement': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                       'learn_rate': [0.05, 0.08, 0.1, 0.25, 0.35, 0.5, 1],
                       'learn_rate_annealing': [0.9, 0.93, 0.95, 0.99, 0.1],
                       'col_sample_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                       'sample_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                       'col_sample_rate_change_per_level': [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1,
                                                            1.5, 1.8, 2],
                       'col_sample_rate_per_tree': [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1],
                       'histogram_type': ["UniformAdaptive", "QuantilesGlobal", "Random"]
                       }

    search_criteria = {'strategy': "RandomDiscrete",
                       'stopping_metric': "lift_top_group",
                       'stopping_tolerance': 0.01,
                       'stopping_rounds': 3,
                       'max_runtime_secs': 800,
                       'max_models': 50}

    gbm_grid = H2OGridSearch(H2OGradientBoostingEstimator(seed=1234,
                                                          balance_classes=True,
                                                          max_depth=max_depth,
                                                          nfolds=nfolds),
                             hyperparameters,
                             search_criteria=search_criteria)

    gbm_grid.train(x=selected, y=target, training_frame=train, validation_frame=test)

    gbm_grid_table = gbm_grid.get_grid(sort_by='auc', decreasing=True)
    gbm_best_model = gbm_grid.models[0]
    gbm_best_model.name = 'best GBM - grid search'
    gbm_grid_table = gbm_grid_table.sorted_metric_table().drop('model_ids', axis=1)[0:show_top]

    return gbm_best_model, gbm_grid_table


def glm_grid_search(selected, target, train, test, nfolds, show_top):
    """
    Performs grid search on a GLM model to find the optimal parameters that maximize the AUC on the test frame.
    Returns the best model and the grid. 

    :param selected: list of selected variables
    :param target: target variable
    :param train: h2o frame with train data
    :param test: h2o frame with test data
    :param nfolds: the number of folds to use for cross-validation
    """
    hyperparameters = {'alpha': [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1],
                       'lambda': [0, 1, 0.5, 0.1, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]}

    search_criteria = {'strategy': "Cartesian"}

    glm_grid = H2OGridSearch(H2OGeneralizedLinearEstimator(family='binomial',
                                                           seed=1234,
                                                           balance_classes=True,
                                                           nfolds=nfolds,
                                                           standardize=True),
                             hyperparameters,
                             search_criteria=search_criteria)

    glm_grid.train(x=selected, y=target, training_frame=train, validation_frame=test)

    glm_grid_table = glm_grid.get_grid(sort_by='auc', decreasing=True)
    glm_best_model = glm_grid.models[0]
    glm_best_model.name = 'best GLM - grid search'
    glm_grid_table = glm_grid_table.sorted_metric_table().drop('model_ids', axis=1)[0:show_top]

    return glm_best_model, glm_grid_table

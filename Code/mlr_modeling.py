from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from .utils import SHAP_analysis


def regression_modeling(filename, objective, 
                        further_objectives = None, n_feat = 3,
                        repeats_outer = 5, k_outer = 4, 
                        repeats_inner = 10, k_inner = 5, 
                        fname_shap = "df_shap.csv",
                        feature_cutoff = 20, corr_cutoff = 0.7,
                        fname_pred = "mlr_predictions.csv",
                        print_pred = True,
                        directory = "."):
    """
    Generate a linear regression model to predict the performance of unseen substrates
    using a nested, repeated cross-validation (CV) approach.
    --------------------
    filename: str
        name of the searchspace .csv file with experimental results
    objective: str
        name of the reaction objective to be modelled
        NOTE: Only one objective can be modelled!
    further_objectives: list or None
        list of all other objectives of the reaction
        Default is None --> no further objectives
    n_feat: int 
        (Default = 3)
        maximum number of features in the generated model
    repeats_outer: int 
        (Default = 5)
        repeats of the outer CV loop
    k_outer: int 
        (Default = 4)
        number of folds in the outer CV loop
    repeats_inner: int 
        (Default = 10)
        repeats of the outer CV loop
    k_inner: int 
        (Default = 10)
        number of folds in the outer CV loop
    fname_shap: str 
        (Default = "df_shap.csv")
        name for the file generated for the shap analyis in the feature reduction
    feature_cutoff: int 
        (Default = 20)
        maximum number of features to consider in the inner CV loop
        if there are more features after preprocessing, the top-[value] features in a SHAP 
        analysis of the ScopeBO surrogate model will be kept
    corr_cutoff: float 
        (Default = 0.7)
        cutoff value for the Pearson correlation analysis in the feature preprocessing
        features with correlations to other features higher than the value will be removed
    fname_pred: str
        (Default = "performance_prediction.csv")
        name of the generated file with the model predictions
    print_pred: Boolean
        (Default = True)
        print the file with the predictions
    directory: str (Default = current directory)
        working directory
    --------------------
    Prints the best model and its statistics.
    Returns the predictions.

    """

    print("This might take a moment. Please wait!")

    # read in the search space
    wdir = Path(directory)
    df_exp = pd.read_csv(wdir / filename,index_col=0,header=0, float_precision = "round_trip")

    # drop the priority column if present
    if "priority" in df_exp.columns:
        df_exp = df_exp.drop(labels = "priority", axis = 1)

    # prune to the samples with experimental data
    df = df_exp.loc[df_exp[objective] != "PENDING"].copy()

    # drop the data for other objectives if they exist
    all_obj = [objective]
    if further_objectives is not None:
        df = df.drop(labels = further_objectives, axis = 1)
        all_obj += further_objectives

    # separate the features and labels
    X = df[[col for col in df.columns if col != objective]].copy()
    y = df[objective].copy()
    y = y.astype(float)

    # instantiate object for outer CV
    outer_cv = RepeatedKFold(n_splits = k_outer,
                            n_repeats = repeats_outer,
                            random_state = 42)

    # progress bar for overall CV manifold progress
    pbar  =tqdm (total = outer_cv.get_n_splits() * k_inner * repeats_inner,
                desc = "Model optimization progress")

    # variable to save outer CV results
    out_cv_models = defaultdict(list)

    # loop through the outer folds
    for (train_idx_out, test_idx) in outer_cv.split(X, y):
        # split the data for the outer CV
        X_train_out = X.iloc[train_idx_out].copy()
        X_test = X.iloc[test_idx].copy()
        y_train_out = y.iloc[train_idx_out].copy()
        y_test = y.iloc[test_idx].copy()

        # instantiate object for inner CV
        inner_cv = RepeatedKFold(n_splits = k_inner,
                            n_repeats = repeats_inner,
                            random_state = 42)
        
        # variable to save the inner CV results
        inn_cv_models = defaultdict(list)


        # loop through the inner CV
        for (train_idx_inn, val_idx) in inner_cv.split(X_train_out, y_train_out):

            # split the data for the inner CV
            X_train_inn = X_train_out.iloc[train_idx_inn].copy()
            X_val = X_train_out.iloc[val_idx].copy()
            y_train_inn= y_train_out.iloc[train_idx_inn].copy()
            y_val = y_train_out.iloc[val_idx].copy()

            # removely correlated features by correlation analysis of inner train
            corr_mat = X_train_inn.corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k = 1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > corr_cutoff)]
            X_train_inn = X_train_inn.drop(to_drop, axis = 1)
            features = X_train_inn.columns.to_list()

            # if there are more features left than the feature_cutoff, 
            # prune to the top [feature_cutoff] most important feature in a SHAP analysis
            if len(features) > feature_cutoff:
                features = _SHAP_pruning(X=X_train_inn, df_exp=df_exp, curr_feat=features,
                                        all_obj=all_obj, fname_shap=fname_shap, 
                                        feat_cutoff=feature_cutoff, wdir=wdir)

            # prune inner train and val to the preprocessed features
            X_train_inn = X_train_inn[features]
            X_val = X_val[features]

            # scale the features for inner train
            scaler = StandardScaler()
            X_train_inn_sc = pd.DataFrame(scaler.fit_transform(X_train_inn), columns = features)

            # apply the same scaling to inner val
            X_val_sc = pd.DataFrame(scaler.transform(X_val), columns = features)

            # test all feature combinations (up to 3 features) on the inner train
            # keep the best feature combinations (returns tuples of comb and RSS)
            inn_train_models = _feature_search(X_train_inn_sc, y_train_inn, features, n_feat)

            # evaluate the top models on the inner validation data
            for comb, _ in inn_train_models:
                model = LinearRegression().fit(X_train_inn_sc[comb], y_train_inn)
                y_val_pred = model.predict(X_val_sc[comb])
                rsme = np.sqrt(mean_squared_error(y_val, y_val_pred))
                # record the features and rmse
                inn_cv_models[tuple(comb)].append(rsme)

            # current inner fold finished - update tqdm
            pbar.update(1)

        # average rsme for each feature comb selected by the inner cv models
        best_inn_models = {k: np.mean(v) for k, v in inn_cv_models.items()}

        # get the features with lowest mean rsme (inner CV champion)
        best_inn_model_feat = list(min(best_inn_models, key=best_inn_models.get))

        # fit a scaler on the outer train and scale it
        scaler = StandardScaler()
        X_train_out_sc = pd.DataFrame(scaler.fit_transform(X_train_out), 
                                    columns = X_train_out.columns)

        # apply the same scaling to outer test
        X_test_sc = pd.DataFrame(scaler.transform(X_test), 
                                columns = X_test.columns)

        # refit model using best feature subset on outer train
        model = LinearRegression().fit(X_train_out_sc[best_inn_model_feat], y_train_out)

        # evaluate on outer train
        y_train_out_pred = model.predict(X_train_out_sc[best_inn_model_feat])
        train_rmse = np.sqrt(mean_squared_error(y_train_out, y_train_out_pred))
        r2_train = _calculate_r2(y_train_out, y_train_out_pred)
        adj_r2_train = _calculate_adjusted_r2(r2_train,
                                            len(y_train_out),
                                            len(best_inn_model_feat))

        # evaluate on outer test
        y_test_pred = model.predict(X_test_sc[best_inn_model_feat])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_test = _calculate_r2(y_test, y_test_pred)
        adj_r2_test = _calculate_adjusted_r2(r2_test,
                                            len(y_train_out),
                                            len(best_inn_model_feat))

        out_cv_models[tuple(best_inn_model_feat)].append((train_rmse, r2_train, adj_r2_train,
                                                        test_rmse, r2_test, adj_r2_test))

    # average the performance for all outer CV models
    best_out_models = {k: tuple(np.mean(score) for score in zip(*v)) for 
                       k,v in out_cv_models.items()}

    # pick the model that was picked most often
    highest_counter = max(len(v) for v in out_cv_models.values())
    most_freq = [k for k,v in out_cv_models.items() if len(v) == highest_counter]
    champion_feat = None
    if len(most_freq) == 1:
        champion_feat = most_freq[0]
    else:  # test_rsme tie-breaker
        rsme_vals = [best_out_models[mod][3] for mod in most_freq]  # test_rsme is 4th tuple entry
        top_idx = rsme_vals.index(min(rsme_vals))
        champion_feat = most_freq[top_idx]
    champion_feat = list(champion_feat)

    # train model on the full data (using both scaled and natural features)
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X), 
                                columns = X.columns)
    model = LinearRegression().fit(X_sc[champion_feat], y)

    model_settings = {f: c for f, c in zip(champion_feat, model.coef_)}
    model_settings["Intercept"] = model.intercept_

    print("Model training completed.\n\nHere are the model parameters using scaled features:")

    for k, v in model_settings.items():
        print(f"{k}: {v}")

    print("\nModel statistics (on outer CV splits):")
    print("Mean RMSE:", best_out_models[tuple(champion_feat)][3])
    print("Mean R2 value:", best_out_models[tuple(champion_feat)][4])

    # use the champion model to predict on the full search space

    # drop all objectives, but save the data for the obj to be predicted
    exp_labels = df_exp[objective].to_list()
    df_exp = df_exp.drop(labels = all_obj, axis = 1)

    # apply the scaler
    df_exp_sc = pd.DataFrame(scaler.transform(df_exp), columns = df_exp.columns)

    # predict
    pred_vals = model.predict(df_exp_sc[champion_feat])

    # report the predictions
    df_pred = pd.DataFrame(np.nan, index = df_exp.index, columns = [objective, f"{objective}_pred"])
    df_pred[f"{objective}_pred"] = pred_vals
    df_pred[objective] = exp_labels

    if print_pred:
        # save the predictions
        df_pred.to_csv(wdir / fname_pred, index = True, header = True)
        print(f"\nPredictions have been saved in the file '{fname_pred}' in the folder '{directory}'.")

    else:
        print("Predictions obtained!")

    return df_pred, model_settings


def _SHAP_pruning(X, df_exp, curr_feat, all_obj, fname_shap, feat_cutoff, wdir):
    """
    Prune the number for feature to the most important features in a SHAP
    analysis of the ScopeBO surrogate model (using on the inner train samples)
    """

    # limit the experimental data to the inner train data 
    # and the downselected features + objectives
    df_shap = df_exp.loc[X.index, curr_feat + all_obj]

    # the SHAP analysis requires a csv file as input - generate it
    df_shap.to_csv(wdir / fname_shap, index = True, header = True)

    # do the SHAP analysis and prune the features to the top most important SHAP features
    _, mean_abs_shap = SHAP_analysis(objectives = all_obj, 
                                        filename = fname_shap, plot_type = [])
    features = mean_abs_shap.index.to_list()[:feat_cutoff]

    return features


def _eval_feat_comb(X, y, feat):
    """
    Evaluate a feature combination by building a MLR and 
    returning the residual sum of squares (RSS).

    X, y: features and labels
    feat: list of feature names
    """

    # limit to the selected features
    X_feat = X.loc[:,feat]
    y = np.array(y)

    # build the model
    model = LinearRegression()
    model.fit(X_feat, y)
    y_pred = model.predict(X_feat)

    # return RSS
    return np.sum((y_pred - y)**2)


def _feature_search(X, y, feat, n_feat = 3, top_n = 5):
    """
    Exhaustive search for best feature combinations for a MLR 
    model with up to n_features. Returns the features and RSS
    for the top_n models.
    """

    # Initiate list of combinations with all single features
    combs = [[f] for f in feat]

    # add combinations of up to n_features length (starting from 2 feat)
    for i in range(2, n_feat +1):
        combs += [list(c) for c in combinations(feat, i)]

    search_results = []
    for i,comb in enumerate(combs):
        rss = _eval_feat_comb(X, y, comb)
        search_results.append((comb,rss))

    # sort by rss and keep the top_n models
    search_results.sort(key = lambda x: x[1])
    top_models = search_results[:top_n]

    return top_models


def _calculate_r2(y_true, y_pred):
    """Calculate R² using the squared Pearson correlation coefficient."""
    correlation_matrix = np.corrcoef(y_true, y_pred)
    r = correlation_matrix[0, 1]
    return r**2


def _calculate_adjusted_r2(r_squared, n_samples, n_features):
    """Adust R2 value for number of samples and features."""
    return 1 - (1 - r_squared) * ((n_samples - 1) / (n_samples - n_features - 1))
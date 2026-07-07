import os
from pathlib import Path
import random
import sys

from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling.samplers import SobolQMCNormalSampler
from IPython.display import display
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import shap
from sklearn.preprocessing import MinMaxScaler
import torch
from vendi_score import vendi

from .model import get_covar_matrix, build_and_optimize_model


def obtain_full_covar_matrix(objectives, directory, filename):
    """
    Calculate the covariance matrix for the full dataset.

    Inputs:
        objectives: list
            list of the objectives. E. g.: ["yield","ee"]
        directory: str
            working directory
        filename: str
            filename of the dataframe
    """

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cpu"),
        }

    wdir = Path(directory)
    csv_filename = wdir.joinpath(filename)

    # Check objectives is a list (even for single objective optimization).
    if type(objectives) != list:
        objectives = [objectives]       

    # Load reaction space from scope csv file and remove columns without any values.
    df = pd.read_csv(f"{csv_filename}",index_col=0,header=0, float_precision = "round_trip")
    df = df.dropna(axis='columns', how='all')
   
    # Prepare the data for the calculation of the covariance matrix by removing objectives and priority columns.
    if 'priority' in df.columns.tolist():
        df = df.drop(columns=['priority'])
    for objective in objectives:
        if objective in df.columns.tolist():
            df = df.drop(columns = objective)

    # Sort the df by index to get consistent order for all the subsequent uses of the matrix.
    sorted_df = df.sort_index()

    # Scaling of input data and conversion to tensor.
    scaler_x = MinMaxScaler()
    scaler_x.fit(sorted_df.to_numpy())
    df_np = scaler_x.transform(sorted_df.to_numpy())  
    df_torch = torch.tensor(df_np.tolist()).to(**tkwargs).double()
    
    # Calculate the covariance matrix of the prior for the full data and normalize it.
    # The covariance matrix needs to be normalized so that that the similarity values k for all entries x are k(x,x) = 1. 
    # This is a requirement for the vendi score calculation.
    covariance_matrix = pd.DataFrame(get_covar_matrix(df_torch).numpy())  # Calculate matrix object and convert it to df
    cov_min = covariance_matrix.min().min()
    cov_max = covariance_matrix.max().max()
    covariance_matrix_scaled = covariance_matrix.applymap(lambda x: (x-cov_min)/(cov_max-cov_min))

    return covariance_matrix_scaled


def calculate_vendi_score(idx_num, covariance_matrix):

    """
    Calculates the Vendi score for all samples that have been evaluated so far (= training data).
    To do this, the covariance matrix of the surrogate model prior is used as the similarity matrix.
    ------------------------------------------------------------------------------
    Input:
        idx_num: list
            list of the (numeric) indices for which the vendi score will be calculated.
        covariance_matrix: df
            symmetric covariance matrix for the entire dataset with diagonal entries
            having the value 1.
    -------------------------------------------------------------------------------
    Returns the vendi score for the set containing previously measured entries.
    """

    # Calculate the vendi score for the subset of interest.
    current_vendi_score = vendi.score_K(covariance_matrix.loc[idx_num, idx_num].to_numpy())

    return current_vendi_score


def vendi_pruning(idx_test, idx_train, Vendi_pruning_fraction, cumulative_test_x,
                  cut_by_vendi, full_covariance_matrix, df, seed):
     
    """
    Prunes a percentage of the test samples from consideration based on the Vendi scores
    that they have when added to the training samples. Lower Vendi scores indicate lower diversity.
    -----------------------------------------------------------------
    Inputs:
        idx_test: list
            list of indices of the test set in the Dataframe df
        idx_train: list
            list of indices of the training data in the DataFrame df
        Vendi_pruning_fraction: float
            percentage to which the test set will be pruned by the Vendi scoring
        cumulative_test_x: list
            list containing the scaled test data
        cut_by_vendi: list
            list containing the indices of previously cut samples.
        full_covariance_matrix: df
            covariance matrix for the full dataset
        df: df
            dataframe for the dataset
        seed: int
            random seed for reproducibility
    ------------------------------------------------------------------
    Returns the modified variables idx_test and cumulative_test_x as well as the list
    of the indices of the pruned samples (cut_by_vendi).
    """

    random.seed(seed)

    # Sort the df for consistent indexing for the covariance matrix.
    sorted_df = df.sort_index()

    # Calculate the Vendi scores.
    vendi_scores = []
    for ind in idx_test:
        idx_combination = np.concatenate((idx_train,np.array([ind])),axis=0)  # indices of training data + point to be evaluated
        idx_comb_num = [sorted_df.index.get_loc(idx) for idx in idx_combination]  # convert to numeric indices
        current_score = calculate_vendi_score(idx_num=idx_comb_num, covariance_matrix=full_covariance_matrix) # calculate Vendi score
        vendi_scores.append(current_score)

    # Prune the reaction space based on Vendi score. Pruned samples are saved.

    # sort the Vendi scores
    sorted_vendi = vendi_scores.copy()
    sorted_vendi.sort(reverse=True)

    # determine the lowest scores that will be cut off from the test set
    cutoff = int(len(sorted_vendi)*((100-Vendi_pruning_fraction)/100))
    cut_vendi_scores = []
    while len(sorted_vendi) > cutoff:
        cut_score = sorted_vendi.pop()
        cut_vendi_scores.append(cut_score)

    # determine the list indices that belong to these vendi scores
    list_positions_cut_vendi = []
    for j in range(len(cut_vendi_scores)):
        position = [i for i,x in enumerate(vendi_scores) if x == cut_vendi_scores[j]]
        # add the positions of all list occurances of the cut vendi scores to the position list
        remaining_samples = len(cut_vendi_scores)-len(list_positions_cut_vendi)
        if remaining_samples != 0:
            if len(position) <= remaining_samples:
                for k in range(len(position)):
                    # Avoid adding a sample multiple times if there are multiple samples with the same vendi score
                    if position[k] not in list_positions_cut_vendi:
                        list_positions_cut_vendi.append(position[k])
            else:
                # Only add samples until cutoff value is reached  --> pick random subset of the samples with the Vendi score at the threshold
                selected_positions = random.sample(position,remaining_samples)
                for k in range(len(selected_positions)):
                    if selected_positions[k] not in list_positions_cut_vendi:
                        list_positions_cut_vendi.append(selected_positions[k])
        else:
            break
    
    # remove the corresponding points from the test dataset and test index list
    list_positions_cut_vendi.sort(reverse=True)
    for i in range(len(list_positions_cut_vendi)):
        cumulative_test_x.pop(list_positions_cut_vendi[i])
        cut_by_vendi.append(idx_test[list_positions_cut_vendi[i]])
    idx_test = np.delete(idx_test,list_positions_cut_vendi)

    return cumulative_test_x, cut_by_vendi, idx_test


def variance_pruning(idx_test,n_objectives,Vendi_pruning_fraction,cumulative_test_x,
                     cumulative_train_x, cumulative_train_y, cut_by_vendi):
     
    """
    Prunes the test set based on the provided Vendi_threshold.
    Only impolemented for benchmarking purposes and single objective optimization.
    -----------------------------------------------------------------
    Inputs:
        idx_test: list
            list of indices of the test set in the Dataframe df
        n_objectives: int
            number of objectives
        Vendi_pruning_fraction: int
            percentage to which the test set will be pruned by the Vendi scoring
        cumulative_test_x: list
            list containing the scaled test data
        cumulative_train_x: list
            list containing the scaled training data
        cumulative_train_y: list
            list containing the scaled training outputs
        cut_by_vendi: list
            list containing the indices of previously cut samples.
    ------------------------------------------------------------------
    Returns the modified variables idx_test and cumulative_test_x as well as the list
    of the indices of the pruned samples (cut_by_vendi).
    """

    tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
    }

    # First, the surrogate model is trained.

    # Tensors for input data.
    train_x_torch = torch.tensor(cumulative_train_x).to(**tkwargs).double()
    test_x_torch = torch.tensor(cumulative_test_x).double().to(**tkwargs)
    
    surrogate_model = None
    individual_models = []
    # Create GP models for each objective.
    for i in range(0, n_objectives):
        
        # Convert training outputs for the objective in question to tensors.
        train_y_i = np.array(cumulative_train_y)[:, i]
        train_y_i = (np.atleast_2d(train_y_i).reshape(len(train_y_i), -1))
        train_y_i_torch = torch.tensor(train_y_i.tolist()).to(**tkwargs).double()

        # Optimize GP model using function from model.py.
        gp, likelihood = build_and_optimize_model(train_x=train_x_torch, train_y=train_y_i_torch)

        # Creating a single task GP model for the objective and storing it in individual_models list.
        model_i = SingleTaskGP(train_X=train_x_torch, train_Y=train_y_i_torch,
                            covar_module=gp.covar_module, likelihood=likelihood)
        individual_models.append(model_i)
        gp = []  # empty GP model again before next loop (saves memory)
    
    # set the surrogate model dependent on the number of objectives
    if len(individual_models) > 1:
        surrogate_model = ModelListGP(*individual_models)
    else:
        surrogate_model = individual_models[0]

    # Calculate the variance.
    variance = surrogate_model.posterior(test_x_torch).variance.detach().numpy()

    # sort the posterior variance
    sorted_variance = variance.tolist().copy()
    sorted_variance.sort(reverse=True)  # descending order
 
    # remove the samples with very low variance (end of the list).
    cutoff = int(len(sorted_variance)*((100-Vendi_pruning_fraction)/100))
    cut_variance_scores = []
    while len(sorted_variance) > cutoff:
        cut_score = sorted_variance.pop()
        cut_variance_scores.append(cut_score)

    # determine the list indices that belong to these variance scores
    list_positions_cut_variance = []
    for j in range(len(cut_variance_scores)):
        position = [i for i,x in enumerate(variance) if x == cut_variance_scores[j]]
        # add the positions of all list occurances of the cut vendi scores to the position list
        remaining_samples = len(cut_variance_scores)-len(list_positions_cut_variance)
        if remaining_samples != 0:
            if len(position) <= remaining_samples:
                for k in range(len(position)):
                    # Avoid adding a sample multiple times if there are multiple samples with the same vendi score
                    if position[k] not in list_positions_cut_variance:
                        list_positions_cut_variance.append(position[k])
            else:
                # Only add samples until cutoff value is reached  --> pick random subset of the samples with the Vendi score at the threshold
                selected_positions = random.sample(position,remaining_samples)
                for k in range(len(selected_positions)):
                    if selected_positions[k] not in list_positions_cut_variance:
                        list_positions_cut_variance.append(selected_positions[k])
        else:
            break
    
    # if there are duplicate vendi scores, the corresponding list positions would be added multiple times - filter duplicates using set()
    list_positions_cut_variance = list(set(list_positions_cut_variance))

    # remove the corresponding points from the test dataset and test index list
    list_positions_cut_variance.sort(reverse=True)  # descending order
    for position in list_positions_cut_variance:
        cumulative_test_x.pop(position)
        cut_by_vendi.append(idx_test[position])
    idx_test = np.delete(idx_test,list_positions_cut_variance)

    return cumulative_test_x, cut_by_vendi, idx_test


def SHAP_analysis(filename,
                  objectives = None,
                  objective_mode = {"all_obj":"max"},  
                  plot_type = ["bar"],
                  directory = "."):
    """
    Analyzes the importance of features on the surrogate model using SHAP.
    ---------------------------------------------------------------------
    Inputs:
        filename: str
            filename of the reaction space csv file including experimental outcomes
        objectives: list or None
            list of the objectives. E. g.: [yield,ee]
            If None (default): will try to infer the objectives by looking for columns with the value "PENDING"
        objective_mode: dict
            Dictionary of objective modes for objectives
            Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
            Code will assume maximization for all non-listed objectives
            Default is {"all_obj":"max"} --> all objectives are maximized   
        plot_type: list of str
            type of SHAP plot to be generated. Options:
                "bar" - bar plot of mean absolute SHAP values (Default)
                "beeswarm" - beeswarm plots of the individual SHAP values
                both options can be requested by using plot_type = ["bar","beeswarm"]
        directory: str
            Define working directory. Default is current directory.
    ---------------------------------------------------------------------
    Returns the shap.explainer object, mean absolute SHAP values,
    and prints the requested plot of the SHAP values.
    """

    tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
    }

    surrogate_model, df_train_x_scaled, _ = _surrogate_preparation(filename=filename,
                                             objectives=objectives,
                                             objective_mode=objective_mode,
                                             directory=directory)

    def _predict_fn(X):
        """
        Creates a suitable input for a shap.Explainter from a gaussian procress surrogate model.
        ----------------------------------------------------------------------------------------
        X: df
            Dataframe of the scaled training features.
        ----------------------------------------------------------------------------------------
        Returns a numpy array with the model predictions.
        """
        X_tensor = torch.tensor(X.to_numpy()).to(**tkwargs).double()  # Ensure the input is a tensor
        with torch.no_grad():
            # Get the mean prediction from the model's posterior distribution
            return surrogate_model.posterior(X_tensor).mean.detach().numpy()

    # Create the SHAP explainer and calculate SHAP values.
    explainer = shap.Explainer(_predict_fn,df_train_x_scaled)
    max_evals = 500
    if len(df_train_x_scaled.columns) > 249:
        max_evals = 2 * len(df_train_x_scaled.columns) + 1 
    shap_values = explainer(df_train_x_scaled, max_evals=max_evals)

    # Process the SHAP values to get mean absolute SHAP values for each feature.
    df_shap = pd.DataFrame(shap_values.values,columns=df_train_x_scaled.columns)
    df_shap = df_shap.applymap(lambda x: abs(x))
    mean_abs_shap_values = df_shap.mean()
    mean_abs_shap_values.sort_values(ascending=False)

    if "bar" in plot_type:
        shap.plots.bar(shap_values)
    if "beeswarm" in plot_type:
        shap.plots.beeswarm(shap_values)

    return shap_values, mean_abs_shap_values.sort_values(ascending=False)


def exp_imp_calc(filename="reaction_space.csv",
                 objectives = None,
                 objective_mode = {"all_obj": "max"},
                 pred_for_cut = False,
                 directory = "."):
    """
    Calculates the predicted mean, variance, and expected improvement for all test substrates
    from a ScopeBO().run() output csv file.
    ---------------------------------------------------------------------
    Inputs:
        filename: str
            filename of the reaction space csv file including experimental outcomes
        objectives: list or None
            list of the objectives. E. g.: [yield,ee]
            If None (default): will try to infer the objectives by looking for columns with the value "PENDING"
        objective_mode: dict
            Dictionary of objective modes for objectives
            Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
            Code will assume maximization for all non-listed objectives
            Default is {"all_obj":"max"} --> all objectives are maximized
        pred_for_cut: Boolean
            Default = False
            Show predictions also for cut samples if True
        results_filename: str or None
            if provided, saves the results dataframe as a csv file under this name
        directory: str
            Define working directory. Default is current directory.
    ---------------------------------------------------------------------
    Returns:
    Dataframe with the compound name as index and the predicted mean, variance, and expected 
    improvement as columns.
    Also saves this data as a csv file if results_filename is not None.
    """
    
    tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
    }

    surrogate_model, df_train_x_scaled, (df_test_x_scaled, df_train_y_scaled, scaler_y) = _surrogate_preparation(filename=filename, 
                                                                                  objectives=objectives,
                                                                                  objective_mode=objective_mode,
                                                                                  pred_for_cut = pred_for_cut,
                                                                                  directory=directory)

    cumulative_train_y = df_train_y_scaled.to_numpy().tolist()

    train_x_torch = torch.from_numpy(df_train_x_scaled.astype(float).to_numpy()).to(**tkwargs)
    test_x_torch = torch.from_numpy(df_test_x_scaled.astype(float).to_numpy()).to(**tkwargs)
    
    sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True, seed=42)

    if len(df_train_y_scaled.columns) > 1:  # multiple objectives
        # determine reference point
        ref_mins = np.min(cumulative_train_y, axis=0)
        ref_point = torch.tensor(ref_mins).double().to(**tkwargs)

        # Generate acquisition function object. 
        # Warnings ignored because it automatically generated a non-consequential numerical warning 
        # for added jitter of ca. 10^-8 otherwise.
        with HiddenPrints():  
            acquisition_function = qNoisyExpectedHypervolumeImprovement(
                model=surrogate_model, sampler=sampler,
                ref_point=ref_point, alpha = 0.0,
                incremental_nehvi = True, X_baseline=train_x_torch, prune_baseline=True
            )

    else:  # Acquisition function for single-objective optimization (qEI)
        # Generate acquisition function object. Warnings ignored because it automatically generated 
        # a numerical warning for added jitter of ca. 10^-8 otherwise.
        with HiddenPrints():
            train_y_torch = torch.tensor(cumulative_train_y).to(**tkwargs).double()
            best_value = train_y_torch.max()
            acquisition_function = qExpectedImprovement(
                model = surrogate_model, 
                best_f = best_value,
                sampler = sampler
                )
            
    # get the surrogate model mean values
    mean = surrogate_model.posterior(test_x_torch).mean.detach().numpy()
    # the mean values are standardized --> rescale to natural values
    natural_mean = scaler_y.inverse_transform(mean)

    # get the surrogate model variance values
    variance = surrogate_model.posterior(test_x_torch).variance.detach().numpy()
    # the variance values are standardized --> rescale to natural values
    natural_variance = scaler_y.inverse_transform_var(variance)
    # convert the standard deviation
    std_dev = np.sqrt(natural_variance)

    # get the expected (hypervolume) improvement values
    exp_imp = acquisition_function(test_x_torch.unsqueeze(1)).detach().numpy()

    objectives =df_train_y_scaled.columns.to_list()

    exp_label = "Expected improvement" if len(objectives) == 1 else "Expected hypervolume improvement"

    # set up a dataframe for the results
    results_data = pd.DataFrame(np.nan, 
                                index = df_test_x_scaled.index.to_list(),
                                columns = [name for obj in objectives
                                           for name in (f"Prediction_{obj}", f"Std. dev. of pred._{obj}")]
                                           + [exp_label])

    # fill in the data
    results_data[exp_label] = exp_imp
    for i,obj in enumerate(objectives):
        # convert the sign of minimization obj back to the orginal state
        if (obj in objective_mode):
            if objective_mode[obj] == "min":  
                natural_mean[:,i] = - natural_mean[:,i]
        results_data[f"Prediction_{obj}"] = natural_mean[:,i]
        results_data[f"Std. dev. of pred._{obj}"] = std_dev[:,i]

    # sort by predicted mean
    results_data.sort_values(by = f"Prediction_{objectives[0]}", axis = 0, 
                             ascending =False, inplace = True)

    return results_data


def draw_suggestions(df):
    """
    Extracts the suggested samples (priority = 1) as well as alternative suggestions (0 < priority < 1)
    and draws their structure.

    df: DataFrame
        DataFrame containing a priority list of suggested molecules
    """

    def _drawing_function(smiles_list):
        """
        Draws the molecules provided in a list of smiles string.
        """
        smiles_list = [str(entry.encode().decode('unicode_escape')) for entry in smiles_list]

        try:
            mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            # Generate 2D coordinates for all mols
            for mol in mol_list:
                AllChem.Compute2DCoords(mol)

            # Draw the aligned molecules
            depiction = Draw.MolsToGridImage(
                mol_list,
                molsPerRow=len(mol_list),
                subImgSize=(200, 200),
                legends=[smiles for smiles in smiles_list]
                )
            display(depiction)
        except:
            print(f"Could not draw the molecules in {smiles_list}.")
            print(f"Please label your molecules with SMILES strings to draw the results of the run.")

    # Extract the suggested molecules.
    suggestions = df[df["priority"] == 1].index.to_list()
    # Extract alternative suggestions (fractional priority) if there are any.
    alternative_suggestions = df[(df["priority"] > 0) & (df["priority"] < 1)].index.to_list()

    # Check if this is a multi-component reaction (substrates separated by ".").
    if "." in suggestions[0]:  # multi-component reaction
        print("These are the suggested substrate combinations:")
        for suggestion in suggestions:
            _drawing_function(smiles_list=suggestion.split("."))
        if alternative_suggestions:
            print("These are the requested alternative suggestions, sorted by descending priority:")
            for alternative_suggestion in alternative_suggestions:
                _drawing_function(smiles_list=alternative_suggestion.split("."))

    else:  # single-component reaction
        print("These are the suggested substrates:")
        _drawing_function(smiles_list=suggestions)
        if alternative_suggestions:
            print("These are the requested alternative suggestions, sorted by descending priority:")
            _drawing_function(smiles_list=alternative_suggestions)


def _surrogate_preparation(filename,
                  objectives = None,
                  objective_mode = {"all_obj":"max"},
                  pred_for_cut =False, 
                  directory = "."):
    """
    Function to prepare a surrogate model from a ScopeBO().run() output csv file.
    Used in SHAP_analysis and expected_improvement_calc functions.
    
    See the doc string of these functions for input parameters.

    Returns the surrogate model object as SingleTaskGP (1 objective) or ModelListGP (multiple objectives).
    Returns the scaled training. 
    Also scaled test features as well as the scaled training labels and the label scaler object as tuple.
    """

    tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
    }

     # Load the data
    wdir = Path(directory)
    df = pd.read_csv(wdir.joinpath(filename),index_col=0,header=0, float_precision = "round_trip")

    # identify the objectives (containing PENDING entries) if none are given
    if objectives is None:
        objectives = df.columns[df.eq("PENDING").any()].to_list()

    # get the data with labels and convert it to the require datatype.
    idx_train = (df[~df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]).index.values

    # test data is data without experimental results (= rows containing "PENDING") and which was not pruned (priority != -1)
    df_noexperiments = df[df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]
    idx_test = None
    # remove the priority column in case that predictions are also desired for cut samples
    if pred_for_cut & ("priority" in df_noexperiments.columns.to_list()):
        df_noexperiments = df_noexperiments.drop(columns=["priority"])
    if "priority" not in df_noexperiments.columns.to_list():
        idx_test = df_noexperiments.index  # assume no pruned samples if no priority column
    else:
        idx_test = df_noexperiments[df_noexperiments['priority'] != -1].index

    # prepare x and y data for BO model by removing objectives and priority columns for the BO model inputs
    df_train_y = df.loc[idx_train][objectives]
    if 'priority' in df.columns.to_list():
        df = df.drop(columns=['priority'])
    df = df.drop(columns=objectives)
    df_train_x = df.loc[idx_train]
    df_test_x = df.loc[idx_test]


    # Scaling of training features and conversion to tensor.
    scaler_x = MinMaxScaler()
    scaler_x.fit(df_train_x.to_numpy())
    train_x_np = scaler_x.transform(df_train_x.to_numpy())


    train_x_torch = torch.tensor(train_x_np.tolist()).to(**tkwargs).double()
    df_train_x_scaled = pd.DataFrame(train_x_np,columns=df_train_x.columns)

    df_test_x_scaled = None
    if len(idx_test) != 0:
        test_x_np = scaler_x.transform(df_test_x.to_numpy())  # transform test features with the same scaler as the train data
        df_test_x_scaled = pd.DataFrame(test_x_np, index = df_test_x.index, columns=df_test_x.columns)

    # Scaling of training outputs. 
    train_y_np = df_train_y.astype(float).to_numpy()
    min_obj = [obj for obj, value in objective_mode.items() if value == "min"]
    if min_obj:
        for obj in min_obj:
            i = objectives.index(obj)
            train_y_np[:, i] = -train_y_np[:, i]

    scaler_y = EDBOStandardScaler()
    train_y_np = scaler_y.fit_transform(train_y_np)
    df_train_y_scaled = pd.DataFrame(train_y_np, index = df_train_y.index, columns=df_train_y.columns)
    cumulative_train_y = train_y_np.tolist()

    # Create GP models for each objective.
    n_objectives = len(objectives)
    individual_models = []
    for i in range(n_objectives):
        
        # Convert training outputs for the objective in question to tensors.
        train_y_i = np.array(cumulative_train_y)[:, i]
        train_y_i = (np.atleast_2d(train_y_i).reshape(len(train_y_i), -1))
        train_y_i_torch = torch.tensor(train_y_i.tolist()).to(**tkwargs).double()

        # Optimize GP model using function from model.py.
        gp, likelihood = build_and_optimize_model(train_x=train_x_torch, train_y=train_y_i_torch)

        # Creating a single task GP model for the objective and storing it in individual_models list.
        model_i = SingleTaskGP(train_X=train_x_torch, train_Y=train_y_i_torch,
                            covar_module=gp.covar_module, likelihood=likelihood)
        individual_models.append(model_i)

    # Define the surrogate model.
    surrogate_model = None
    if n_objectives > 1:
        surrogate_model = ModelListGP(*individual_models)
    else:
        surrogate_model = individual_models[0]

    return surrogate_model, df_train_x_scaled, (df_test_x_scaled, df_train_y_scaled, scaler_y)

      

class EDBOStandardScaler:
    """
    Custom standard scaler taken over from EDBO.
    """
    def __init__(self):
        pass

    def fit(self, x):
        self.mu  = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def inverse_transform(self, x):
        return x * [self.std] + [self.mu]

    def inverse_transform_var(self, x):
        std = np.asarray(self.std)
        return x * (std**2)
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
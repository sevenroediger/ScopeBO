import os
import sys
import warnings
import random

import torch
import numpy as np
import pandas as pd

from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.samplers import SobolQMCNormalSampler
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling, CVTSampling
import itertools
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from .model import build_and_optimize_model
from .space_creator import create_reaction_space
from .utils import EDBOStandardScaler, calculate_vendi_score, obtain_full_covar_matrix, vendi_pruning, variance_pruning, SHAP_analysis, draw_suggestions
from .acquisition import greedy_run, explorative_run, random_run, low_variance_selection


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class ScopeBO:

    def __init__(self):

        self.full_covariance_matrix = None


    @staticmethod
    def generate_features():
        """
        Generates featurization of the reactants using Mordred descriptors.
        We recommend to use DFT features, but if these are not available, this function can be
        used to quickly generate featurization with the computational overhead associated with 
        DFT calculations.
        The function generates the descriptors for all substrates and then uses Pearson
        correlation analysis to remove highly correlated descriptors.
        --------------------------------------------------------------------------------------
        Inputs:
            reactants: list of str
                names of the csv files containing all possible substrates for the respective
                reactant.
        --------------------------------------------------------------------------------------
        Returns:
            No returns, but generates csv files for all reactants with their featurization. 
            These can then be read into the create_reaction_space function.
        """

        # NOTE: write function code. Maybe outsource to other file.
    
    @staticmethod
    def create_reaction_space(reactants, directory='./', filename='reaction_space.csv'):
        """
        Creates a reaction space with all possible scope combinations.
        
        reactants is a list of csv filenames (one file per reactant). The csv files should 
        contain the name of the reactant in the first column and features in the other columns.

        Function create_reaction_scope is in space_generator.py.
        
        Returns a dataframe df with all search space reaction combinations.
        """
        df = create_reaction_space(components=reactants, directory=directory,
                                   filename=filename)
        return df
    

    @staticmethod
    def feature_analysis(objectives,objective_mode,filename,plot_type=["bar"],directory="."):
        """
        Analyzes the importance of features on the surrogate model using SHAP.
        ---------------------------------------------------------------------
        Inputs:
            filename: str
                filename of the reaction space csv file including experimental outcomes
                default is reaction_space.csv
            objectives: list
                list of the objectives. E. g.: [yield,ee]
            objective_mode: list
                list of the mode of the objectives (max or min)
            directory: str
                name of the working directory.
                Default is the current directory.
            seed: int  
                random seed
            plot_type: list of str
                type of SHAP plot to be generated. Options:
                    "bar" - bar plot of mean absolute SHAP values (Default)
                    "beeswarm" - beeswarm plots of the individual SHAP values
                    both options can be requested by using plot_type = ["bar","beeswarm"]
            directory: str
                Define working directory. Default is current directory.

        ---------------------------------------------------------------------
        Returns the shap.explainer object, mean absolute SHAP values, and the requested plot of SHAP values.
        """

        # Call the function from the utils file
        shap_values, mean_abs_shap_values = SHAP_analysis(
            objectives=objectives,
            objective_mode=objective_mode,
            filename=filename,
            plot_type=plot_type,
            directory=directory)

        return shap_values, mean_abs_shap_values

    
    def get_vendi_score(self, objectives, directory='.', filename='reaction_space.csv'):

        """
        Calculates the Vendi score for all samples that have been evaluated so far (= training data).
        To do this, the covariance matrix of the surrogate model prior is used to calculate the vendi score.
        ------------------------------------------------------------------------------
        Input:
            objectives: list
                list containing the objective names as string values
            directory: str
                working directory
                Default: '.' (current directory)           
            filename: str
                filename of the csv output file from a previous ScopeBO run, containing the featurized
                reaction space and objective values
        -------------------------------------------------------------------------------
        Returns the vendi score for the set containing previously measured entries.
        """
        wdir = Path(directory)
        df = pd.read_csv(wdir.joinpath(filename),index_col=0,header=0, float_precision = "round_trip")
        # Sort the df by index to ensure compatibility with the covariance matrix values.
        sorted_df = df.sort_index()
        #get the indices of all datapoints that were evaluated so far. Samples that have not been measured will have "PENDING" as the entry in the objective column and will be ignored.
        idx_target = (sorted_df[~sorted_df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]).index.values
        # Convert the indices to numeric indices.
        idx_num = [sorted_df.index.get_loc(idx) for idx in idx_target]

        # Calculate the covariance matrix for the full dataset if it hasn't been calculated yet
        if self.full_covariance_matrix is None:
            self.full_covariance_matrix = obtain_full_covar_matrix(objectives=objectives,directory=directory,filename=filename)
        # Calculate the vendi score.
        current_vendi_score = calculate_vendi_score(idx_num=idx_num,covariance_matrix=self.full_covariance_matrix)
        return current_vendi_score
    

    @staticmethod
    def _init_sampling(df, batch, seed, sampling_method='cvt'):
        
        """
        Sampling to select the first experiments to suggest.

        -----------------------------------------------------------------------
        Inputs: 
            df: DataFrame 
                Reaction space from generate_reaction_scope
            batch: float
                Number of experiments to suggest.
            sampling_method: String
                Selected sampling method.
                Options:    random
                            lhs (LatinHypercube)
                            cvt (CVTSampling) --> Default
            seed: int
                random seed
        -----------------------------------------------------------------------
        Returns:
            df: original DataFrame with added priorites for suggested scope entries.
        """

        class HiddenPrints:
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout

        # Order df according to initial sampling method (random samples).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with HiddenPrints():
                np.random.seed(seed)
                random.seed(seed)
                # Get sampling points.
                idaes = None
                if sampling_method == 'random':  # Random sampling.
                    samples = df.sample(n=batch, random_state=seed, replace=True)
                elif sampling_method.lower() == 'lhs':
                    # creation of class object.
                    idaes = LatinHypercubeSampling(df, batch, sampling_type="selection")
                elif sampling_method.lower() == 'cvt':
                    idaes = CVTSampling(df, batch, sampling_type="selection")
                
                if idaes is not None:
                    # get the samples.
                    samples = idaes.sample_points()
                
                # Sometimes the LHS or CVT sampling methods return less samples than requested. Add random samples in this case.
                additional_samples = None
                if len(samples) < batch:
                    additional_samples = df.sample(n=batch-len(samples), random_state=seed, replace=True)
                    additional_samples = additional_samples.reset_index(drop=True)
                # Add the additional samples to the samples dataframe. If some of the additional_samples are already in samples, generate new ones until the batch size is reached.
                extra_seed = 45
                while len(samples) < batch:
                    samples = pd.concat([samples,additional_samples]).drop_duplicates(ignore_index=True)
                    additional_samples = df.sample(n=batch-len(samples), random_state=seed+extra_seed, replace=True)
                    extra_seed += extra_seed
                
        # Samples have been created, but need to be assigned a priority and 
        # also to the dataframe.

        # Get index of the best samples according to the random sampling method.
        df_sampling = df.to_numpy()
        priority_list = np.zeros_like(df.index)
        # np.zeros_like creates array of zeros with same shape as given array
        # in this case the indices of df_sampling
        
        # Distance calculation used to figure out which point in df_sampling
        # corresponds to the selected sampling points
        for sample in samples.to_numpy():
            d_i = cdist([sample], df_sampling, metric='cityblock')
            a = np.argmin(d_i)
            priority_list[a] = 1.
        df['priority'] = priority_list # add priorities to the original df
        print(f"Generated {len(samples)} initial samples using {sampling_method} sampling (random seed = {seed}). Run finished!")
        
        return df
    
    def run(self,
            objectives, objective_mode, objective_weights=None,
            directory='.', filename='reaction_space.csv',
            batch=3, init_sampling_method='random', seed=42,
            Vendi_pruning_fraction=12,  # NOTE: change the defaults once optimized.
            pruning_metric = "vendi_sample",
            acquisition_function_mode='greedy',
            give_alternative_suggestions=True,
            show_suggestions=True,
            sample_threshold=None,
            enforce_dissimilarity=False
            ):
        
        """
        Prepares the dataframe for the BO run and then runs it using the
        nested function model_run (defined below this function).
        
        Parameters
        ----------
        objectives: list
            list of strings containing the name for each objective.
            Example:
                objectives = ['yield', 'cost', 'impurity']
        objective_mode: list
            list to select whether the objective should be maximized or minimized.
            Examples:
                A) Example for single-objective optimization:
                    objective_mode = ['max']
                B) Example for multi-objective optimization:
                    objective_mode = ['max', 'min', 'min']
        objective_weights: list
            list of float weights for the scalarization of the objectives 
            only relevant for multi-objective greedy runs, not other acquisition functions
            NOTE: add the final name of the function
            Default: None (objectives will be averaged)
        directory: string
            name of the directory to save the results of the optimization.
            Default is the current directory
        filename: string
            Name of the file to save a *csv* with the priority list. 
            If *get_predictions=True* EDBO+ will automatically save a second 
            file including the predictions (*pred_filename.csv*).
            Default name is 'reaction.csv'.
        columns_features: list
            List containing the names of the columns to be included in the regression model. By default set to
            'all', which means the algorithm will automatically select all the columns that are not in
            the *objectives* list.
        batch: int
            Number of experiments that you want to run in parallel. For instance *batch = 5* means that you
            will run 5 experiments in each EDBO+ run. You can change this number at any stage of the optimization,
            so don't worry if you change  your mind after creating or initializing the reaction scope.
        init_sampling_method: string:
            Method for selecting the first samples in the scope (in absence)  Choices are:
            - 'random' : Random seed (as implemented in Pandas).
            - 'lhs' : LatinHypercube sampling.
            - 'cvtsampling' : CVT sampling (default option) 
        seed: int
            Seed for the random initialization. Default = 42
        Vendi_pruning_fraction: float
            Pruning factor for the Vendi score evaluation.
            Default is 10 (10% of the possible scope entries are pruned.).
        pruning_metric: str
            Metric used for the pruning.
            Options:
                "vendi": pruning by vendi scores before every round of experiments (default).
                "vendi_sample": pruning by vendi score before every sample.
                "variance": pruning by surrogate model variance (implemented only for benchmarking purposes).
        acquisition_function_mode: str
            Choose the acqusition function.
            Options:
                "balanced" (Default): exploration-exploitation trade-off via ExpectedImprovement (1 objective) or NoisyExpectedHypervolumeImprovement (multi-objective)
                "greedy": pure exploitative selection
                "explorative": pure explorative selection
                "random": random selection
        give_alternative_suggestions: Boolean
            Option to get alternative suggestions.
            Default is True.
        """
        
        # Set filenames, random seeds.
        wdir = Path(directory)
        csv_filename = wdir.joinpath(filename)
        torch.manual_seed(seed)
        np.random.seed(seed)
  

        # 1. Safe checks.
        self.objective_names = objectives

        # Check for correct Vendi_pruning_fraction input.
        msg = "Vendi_pruning_fraction must be between 0 (no pruning) and 100 (all samples pruned). Please check your input."
        assert (Vendi_pruning_fraction >= 0 and Vendi_pruning_fraction <= 100), msg

        # Check if objectives is a list (even for single objective optimization).
        if type(objectives) != list:
            objectives = [objectives]
        if type(objective_mode) != list:
            objective_mode = [objective_mode]

        
        # Check if the objective modes were provided correctly
        msg = "Each objective mode must be either 'max' for maximization or 'min' for minimization. Please check your input."
        assert (all(mode.lower() in {"max", "min"} for mode in objective_mode)),msg

        # Assert that the number of objectives and objective modes matches
        msg = "The number of objective modes and objectives does not match. Please check your input."
        assert (len(objectives) == len(objective_mode)),msg

        # Assert that the correct number of weights are given if they are provided
        if objective_weights is not None:
            msg = "The number of objective weights does not match the number of objectives. Please check your input."
            assert (len(objective_weights) == len(objectives)),msg
            # make sure the weights are all floats
            objective_weights = [float(weight) for weight in objective_weights]

        # Check that the reaction space table exists.
        msg = "Reaction space was not found. Please create one and provide it as input (csv file)."
        assert os.path.exists(csv_filename), msg

        # 2. Load reaction space from scope csv file and remove columns without any values.
        df = pd.read_csv(f"{csv_filename}",index_col=0,header=0, float_precision = "round_trip")
        df = df.dropna(axis='columns', how='all')
        original_df = df.copy(deep=True)  # Make a copy of the original data.

        # 2.1. Initialize sampling (only in the first iteration).
        obj_in_df = list(filter(lambda x: x in df.columns.values, objectives))
        # filter out the objectives that are actually in the scope DataFrame.

        # Check whether new objective has been added
        # if there are, add them to the DataFrame and use PENDING as a dummy value.
        for obj_i in self.objective_names:
            if obj_i not in original_df.columns.values:
                original_df[obj_i] = ['PENDING'] * len(original_df.values)
        

        # If there was not an objective column in the data frame prior to
        # adding missing objective column, there are no experimental results
        # and initialization is used for the next run.
        if len(obj_in_df) == 0:
            print(f"There are no results yet. Suggesting experiments by sampling...")
            df = self._init_sampling(df=df, batch=batch, seed=seed,
                                     sampling_method=init_sampling_method)
            # df is now a dataframe containing selected experiments for evaluation
            # based on sampling method.
            original_df['priority'] = df['priority']  # add priority to orginal_df
            # Append objectives.
            for objective in objectives:
                if objective not in original_df.columns.values:
                    original_df[objective] = ['PENDING'] * len(original_df)

            # Sort values and save dataframe.
            original_df = original_df.sort_values('priority', ascending=False)
            original_df = original_df.loc[:,~original_df.columns.str.contains('^Unnamed')]
            original_df.to_csv(csv_filename, index=True,header=True)

            # Draw the suggestions if requested.
            if show_suggestions:
                draw_suggestions(df=original_df)
        
            return original_df

        # check if there are DataFrame entries with experimental results
        idx_experimental_results = (df[df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]).index.values

        if not idx_experimental_results.tolist():
            
            # If there is a priority list but no results, the user should fill in data.
            if 'priority' in df.columns.values:
                # no experimental data exists that can be used to train the model.
                msg = 'Sampling points were already generated, please ' \
                    'insert at least one experimental observation ' \
                    'value and then run again.'
                print(msg)
                return original_df
            
            else:
                # As there is no existing priority list, sampling will be performed.
                print(f"There are no results yet. Suggesting experiments by sampling...")
                df = self._init_sampling(df=df, batch=batch, seed=seed,
                                        sampling_method=init_sampling_method)
                # df is now a dataframe containing selected experiments for evaluation
                # based on sampling method.
                original_df['priority'] = df['priority']  # add priority to orginal_df
                # Append objectives.
                for objective in objectives:
                    if objective not in original_df.columns.values:
                        original_df[objective] = ['PENDING'] * len(original_df)

                # Sort values and save dataframe.
                original_df = original_df.sort_values('priority', ascending=False)
                original_df = original_df.loc[:,~original_df.columns.str.contains('^Unnamed')]
                original_df.to_csv(csv_filename, index=True,header=True)

                # Draw the suggestions if requested.
                if show_suggestions:
                    draw_suggestions(df=original_df)
                
                return original_df
        
        # Check that the search space still contains enough samples. Also reset priority of suggested samples that were not measured.
        if "priority" in df.columns.values:

            df_noexperiments = df[df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]
            idx_test = df_noexperiments[df_noexperiments['priority'] != -1].index
            
            # Check if there are less samples in the search space than the batch size.
            if len(idx_test) < batch:
                if len(idx_test) == 0:
                    print("There no more samples left in the search space.")
                    return original_df
                else:
                    batch = len(idx_test)
                    if batch == 1:
                        print(f"These is only 1 sample left in the search space. The batch size is thus decreased to 1.")
                    else:
                        print(f"There are only {batch} samples lef in the search space.")
                    print(f"The batch size is therefore decreased to {batch}.")

            # Reset the priority of all samples that have not been measured and that were not pruned.
            for idx in idx_test:
                df["priority"].at[idx] = 0

        # calculate the normalized covariance matrix for the full dataset if it has not been calculated yet.
        if self.full_covariance_matrix is None:
            self.full_covariance_matrix = obtain_full_covar_matrix(objectives,directory,filename)
  
        # Run the BO process.
        priority_list = self._model_run(
                df=df,
                batch=batch,
                objectives=objectives,
                objective_mode=objective_mode,
                objective_weights = objective_weights,
                seed=seed,
                Vendi_pruning_fraction=Vendi_pruning_fraction,
                pruning_metric = pruning_metric,
                acquisition_function_mode = acquisition_function_mode,
                full_covariance_matrix = self.full_covariance_matrix,
                give_alternative_suggestions = give_alternative_suggestions,
                sample_threshold=sample_threshold,
                enforce_dissimilarity=enforce_dissimilarity
        )

        # update the priority list in the reaction space dataframe
        original_df["priority"] = priority_list

        # sort the dataframe by priority and save as csv
        original_df.sort_values('priority', ascending=False, inplace=True)
        original_df.to_csv(csv_filename, index=True, header=True)

        print("The run finished sucessfully!")

        # Draw the suggestions if requested.
        if show_suggestions:
            draw_suggestions(df=original_df)

        return original_df


    def _model_run(self, df, batch, objectives, objective_mode, objective_weights,
                   seed, Vendi_pruning_fraction, pruning_metric, acquisition_function_mode,
                   full_covariance_matrix,give_alternative_suggestions,sample_threshold,enforce_dissimilarity):
        """
        Runs the BO process using a Gaussian Process surrogate model and expected improvement-type
        acquisition functions:
            LogNoisyExpectedImprovement for single objective models
            LogNoisyExpectedHypervolumeImprovement for multi objective models

        Returns a priority list for a given reaction space (top priority to low priority).
        
        -------------------------------------------------------
        self: instance of class
        
        df:   Dataframe containing the prepared features for the BO run
                (both test+train, see run function)
        
        batch: int (number of experiments to run in this batch of experiments)

        full_covariance_matrix: DataFrame
            covariance matrix of the full dataset
        
        objective_mode, seed, Vendi_pruning_fraction,
        pruning_metric, give_alternative_suggestions:
            see doc string for run function above.
        """

        class HiddenPrints:
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout
        
        # Separate train and test data.

        # training data is data with experimental results (no element with 'PENDING' value)
        idx_train = (df[~df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]).index.values
        
        # test data is data without experimental results (= rows containing "PENDING") and which was not pruned (priority != -1)
        df_noexperiments = df[df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]
        idx_test = df_noexperiments[df_noexperiments['priority'] != -1].index

        cut_by_vendi = []  # list to hold the experiments cut by the Vendi diversity pruning
        
        # explicitly enforce that the selected points are sufficiently dissimilar from all previously selected points
        if enforce_dissimilarity:
            sorted_df = df.sort_index()
            idx_train_num = [sorted_df.index.get_loc(idx) for idx in idx_train]
            for idx1 in idx_test:
                idx1_num = sorted_df.index.get_loc(idx1)
                too_similar = False
                for idx2_num in idx_train_num:
                    if calculate_vendi_score(idx_num=[idx1_num,idx2_num], covariance_matrix=self.full_covariance_matrix) < 1.06:
                        too_similar = True
                        break
                if too_similar:
                    idx_test = idx_test.drop(idx1)
                    cut_by_vendi.append(idx1)


        # prepare x and y data for BO model by removing objectives and priority columns for the BO model inputs
        df_train_y = df.loc[idx_train][objectives]
        if 'priority' in df.columns.tolist():
            priority_list = list(df["priority"])
            df = df.drop(columns=objectives + ['priority'])
        else:
            df = df.drop(columns=objectives)
        df_train_x = df.loc[idx_train]
        df_test_x = df.loc[idx_test]
        
        # Check number of objectives.
        n_objectives = len(df_train_y.columns.values)

        # Scaling of input data.
        scaler_x = MinMaxScaler()
        scaler_x.fit(df_train_x.to_numpy())
        train_x_np = scaler_x.transform(df_train_x.to_numpy())
        test_x_np = scaler_x.transform(df_test_x.to_numpy())
        
        # Scaling of training outputs.
        train_y_np = df_train_y.astype(float).to_numpy()
        for i in range(0, n_objectives):
            if objective_mode[i].lower() == 'min':
                train_y_np[:, i] = -train_y_np[:, i]
        scaler_y = EDBOStandardScaler()
        train_y_np = scaler_y.fit_transform(train_y_np)


        best_samples = []  # list to hold the suggested experiments
        next_samples = []  # list to hold the next-best experiments as alternatives
        cumulative_train_x = train_x_np.tolist()
        cumulative_train_y = train_y_np.tolist()
        cumulative_test_x = test_x_np.tolist()

        # Set up the Sobol sampler for the acquisition function.
        if len(df.values) > 100000:
            sobol_num_samples = 64
        elif len(df.values) > 50000:
            sobol_num_samples = 128
        elif len(df.values) > 10000:
            sobol_num_samples = 256
        else:
            sobol_num_samples = 512
        sampler = SobolQMCNormalSampler(num_samples=sobol_num_samples, collapse_batch_dims=True, seed=seed)

        # Prune the search space via Vendi scoring if requested.
        if Vendi_pruning_fraction != 0:
            if pruning_metric.lower() == "vendi":
                cumulative_test_x, cut_by_vendi, idx_test = vendi_pruning(
                    idx_test = idx_test,
                    idx_train = idx_train, 
                    Vendi_pruning_fraction = Vendi_pruning_fraction,
                    cumulative_test_x = cumulative_test_x, 
                    cut_by_vendi = cut_by_vendi,
                    full_covariance_matrix=full_covariance_matrix,
                    df=df, seed = seed)
                
            elif pruning_metric.lower() == "variance":  # instead prune by variance (implemented for benchmarking purposes)
                cumulative_test_x, cut_by_vendi, idx_test = variance_pruning(
                    idx_test = idx_test, n_objectives = n_objectives,
                    Vendi_pruning_fraction = Vendi_pruning_fraction,
                    cumulative_test_x = cumulative_test_x, 
                    cumulative_train_x = cumulative_train_x,
                    cumulative_train_y = cumulative_train_y,
                    cut_by_vendi = cut_by_vendi)
    

        # BO with fantasy updates.
        acq_modes_list = ["balanced", "explorative", "greedy", "random"]
        if acquisition_function_mode.lower() in acq_modes_list:

            # Loop through the number of batch samples in the batch.
            for batch_exp in range(batch):

                # Prune the search space via Vendi scoring if requested (this time for each sample (after the fantasies) not each batch).
                if ((pruning_metric.lower()  == 'vendi_sample') and (Vendi_pruning_fraction != 1)):
                    cumulative_test_x, cut_by_vendi, idx_test = vendi_pruning(
                        idx_test = idx_test,
                        idx_train = idx_train, 
                        Vendi_pruning_fraction = Vendi_pruning_fraction,
                        cumulative_test_x = cumulative_test_x, 
                        cut_by_vendi = cut_by_vendi,
                        full_covariance_matrix=full_covariance_matrix,
                        df=df, seed = seed)

                # Instantiate some variables.
                surrogate_model = None
                train_x_torch = None
                test_x_torch = None

                if acquisition_function_mode.lower() != "random":  # no surrogate model needed in case of random selection    
                    # Tensors for input data.
                    train_x_torch = torch.tensor(cumulative_train_x).to(**tkwargs).double()
                    test_x_torch = torch.tensor(cumulative_test_x).double().to(**tkwargs)
                    

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
            
                # Instantiate some variables for the acquisition function evaluation.
                acquisition_function = None 
                sample = None
                samples = None
                idx_sample = None
                list_position_sample = None

                if acquisition_function_mode.lower() == "balanced":
                    if n_objectives > 1:  # Acquisition function for multi-objective optimization (qNoisyEHVI)

                        # Reference point is the minimum seen so far (important for hypervolume calculation).
                        ref_mins = np.min(cumulative_train_y, axis=0)
                        ref_point = torch.tensor(ref_mins).double().to(**tkwargs)

                        # Generate acquisition function object. 
                        # Warnings ignored because it automatically generated a non-consequential numerical warning for added jitter of ca. 10^-8 otherwise.
                        with HiddenPrints():  
                            acquisition_function = qNoisyExpectedHypervolumeImprovement(
                                model=surrogate_model, sampler=sampler,
                                ref_point=ref_point, alpha = 0.0,
                                incremental_nehvi = True, X_baseline=train_x_torch, prune_baseline=True
                            )

                    else:  # Acquisition function for single-objective optimization (qEI)

                        # Generate acquisition function object. Warnings ignored because it automatically generated a numerical warning for added jitter of ca. 10^-8 otherwise.
                        with HiddenPrints():
                            train_y_torch = torch.tensor(cumulative_train_y).to(**tkwargs).double()
                            best_value = train_y_torch.max()
                            acquisition_function = qExpectedImprovement(
                                model = surrogate_model, 
                                best_f = best_value,
                                sampler = sampler
                                )
                    
                    acquisition_samples = 1

                    # Save the next best suggestions in the last round of the batch if requested.
                    if ((batch_exp == batch -1) and (give_alternative_suggestions is True)):
                        # Check that there are enough samples left in the search space.
                        if len(idx_test) < 6:
                            acquisition_samples = len(idx_test)
                        else:
                            acquisition_samples = 6

                    # Optimizes the acquisition function.
                    acq_result = optimize_acqf_discrete(
                        acq_function=acquisition_function,
                        choices=test_x_torch,
                        q=acquisition_samples,
                        unique=True)

                    # Get the samples.
                    samples = acq_result[0].detach().numpy().tolist()

                    for sample in samples:
                        # Find the list position of the selected sample by calculating the distance to points in the test set.
                        sample_dist = None
                        for idx in range(len(cumulative_test_x)):
                            current_dist = np.sum(np.abs(np.array(sample)-np.array(cumulative_test_x[idx])))  # Manhattan distance
                            # Update shortest distance and list index if the current point is closer to the selected sample.
                            if sample_dist is None:
                                sample_dist = current_dist
                                list_position_sample = idx
                            else:
                                if current_dist < sample_dist:
                                    sample_dist = current_dist
                                    list_position_sample = idx

                        # Get the sample index and save it. Update other variables for the batch fantasy.
                        idx_sample = idx_test[list_position_sample]
                        
                        # Save the best sample. Update the batch fantasy.
                        if sample == samples[0]:    
                            idx_test = np.delete(idx_test, list_position_sample)
                            idx_train = np.append(idx_train, idx_sample)
                            popped_sample = cumulative_test_x.pop(list_position_sample)
                            cumulative_train_x.append(popped_sample)
                            best_samples.append(idx_sample)
                        # Save the next best samples.
                        else:
                            next_samples.append(idx_sample)

                    # Reassign the first predicted sample as the best sample for the fantasy prediction below
                    sample = [samples[0]]

                # Fully explorative acquisition function. Implemented for benchmarking purposes and only for single objective runs.
                if acquisition_function_mode.lower() == "explorative":
                    idx_sample, sample, list_position_sample = explorative_run(
                        surrogate_model=surrogate_model, q=1, idx_test=idx_test, 
                        test_x_torch=test_x_torch)
                    
                    # Update batch fantasy.
                    idx_test = np.delete(idx_test, list_position_sample[0])
                    idx_train = np.append(idx_train, idx_sample[0])
                    popped_sample = cumulative_test_x.pop(list_position_sample[0])
                    cumulative_train_x.append(popped_sample)
                    best_samples.append(idx_sample[0])

                # Random sample selection. Implemented for benchmarking purposes.
                if acquisition_function_mode.lower() == "random":
                    idx_sample, list_position_sample = random_run(q=1, idx_test=idx_test,seed=seed)

                    # Update the batch fantasy.
                    idx_test = np.delete(idx_test,list_position_sample[0])
                    idx_train = np.append(idx_train, idx_sample[0])
                    popped_sample = cumulative_test_x.pop(list_position_sample[0])
                    cumulative_train_x.append(popped_sample)
                    best_samples.append(idx_sample[0])

                # Exploitative acquisition function. Only implemented for benchmarking and single objective runs.
                if acquisition_function_mode.lower() == "greedy":
                    
                    acquisition_samples = 1
                    # Save the next best suggestions in the last round of the batch if requested.
                    if ((batch_exp == batch -1) and (give_alternative_suggestions is True)):
                        # Check that there are enough samples left in the search space.
                        if len(idx_test) < 6:
                            acquisition_samples = len(idx_test)
                        else:
                            acquisition_samples = 6

                    # run the acquisition function
                    idx_sample, sample, list_position_sample = greedy_run(
                        surrogate_model=surrogate_model, q=acquisition_samples, objective_mode=objective_mode, idx_test=idx_test, 
                        test_x_torch=test_x_torch)
                    
                    # Update batch fantasy.
                    idx_test = np.delete(idx_test, list_position_sample[0])
                    idx_train = np.append(idx_train, idx_sample[0])
                    popped_sample = cumulative_test_x.pop(list_position_sample[0])
                    cumulative_train_x.append(popped_sample)
                    best_samples.append(idx_sample[0])
                    if len(idx_sample) > 1:
                        for sample in sample[1:]:  # the first entry of the list is the actual suggestion, the rest are the alternatives
                            next_samples.append(sample[i])
   
                if acquisition_function_mode.lower() != "random":  # no prediction needed in case of random acq. fct.
                    # Get the yield prediction for the suggested sample and save it for the fantasy.
                    y_pred = surrogate_model.posterior(
                        torch.tensor(sample)).mean.detach().numpy()[0].tolist()
                    cumulative_train_y.append(y_pred)              

        # selection based on low variance score without batch fantasies (implemented for benchmarking purposes, only single objective possible)
        elif acquisition_function_mode.lower() == "low_variance":
            best_samples = low_variance_selection(batch, idx_test, cumulative_test_x, cumulative_train_x, cumulative_train_y)

        else:
            return print("Acquisition function not found - no samples selected. Please check your input!")

        # remove samples that are too similar to each other in the suggestions if requested
        if (sample_threshold is not None) and (len(best_samples) > 1):

            # Set up variables
            Vendi_cutoff = None
            sample_cutoff = None

            # Define the variables based on user input
            if isinstance(sample_threshold,(int,float)):  # only Vendi cutoff given, no dependence on number of prior samples
                Vendi_cutoff = sample_threshold
            elif isinstance(sample_threshold,tuple):  # cutoffs for Vendi score and number of prior samples are given
                Vendi_cutoff = sample_threshold[0]
                sample_cutoff = sample_threshold[1]
            else:
                print("The input variable must either be numeric or a tuple. Please check your input!")
            
            if (sample_cutoff is None) or (sample_cutoff >= (len(idx_train))-3):  # minus 3 because the three samples from this run have already been added to idx_train

                # calculate the pairwise Vendi scores of the pruned samples and remove those that are too similar to each other
                # generate pairs out of the suggested samples
                sorted_df = df.sort_index()
                pairs = list(itertools.combinations(best_samples, 2))
                dict_pairwise  = [{'sample1': pair[0], 'sample2': pair[1], 
                                'Vendi score': np.nan, 'idx_num1': sorted_df.index.get_loc(pair[0]), 
                                'idx_num2': sorted_df.index.get_loc(pair[1])} for pair in pairs]
                df_pairwise = pd.DataFrame(dict_pairwise)
                
                # calculate the pairwise Vendi scores
                df_pairwise["Vendi score"] = [calculate_vendi_score(idx_num=[df_pairwise.loc[round,"idx_num1"],df_pairwise.loc[round,"idx_num2"]],covariance_matrix=self.full_covariance_matrix) for round in df_pairwise.index]

                # remove the 2nd sample in pairs that have a Vendi score below a certain value (empirically determined as 1.15)
                # as the samples were sorted with decreasing acquisition function priority, the 1st sample has higher priority and is kept
                for idx in set(df_pairwise[df_pairwise["Vendi score"] < Vendi_cutoff]["sample2"]):
                    print(idx)
                    best_samples.remove(idx)
                    idx_train = idx_train[idx_train != idx]  # remove also from idx_train (different command because it is a np.array)
                    cut_by_vendi.append(idx)

        # Next section assigns the samples to the dataframe and creates a priority list.
        
        # add the old priority list again. Samples that have been used this run will be updated.
        df["priority"] = priority_list

        # Assign a very low priority to already observed samples (-2)
        df.loc[df.index.isin(idx_train),"priority"] = -2
        
        # priority -1 if sample was cut by the vendi scoring
        df.loc[df.index.isin(cut_by_vendi),"priority"] = -1

        # Assign high priority to samples selected by the acquisition function (+1)
        df.loc[df.index.isin(best_samples),"priority"] = 1

        # Assign priority 0.x for the 5 next best suggestion; the closer to 1, the higher the sample is on the priority of these samples
        if give_alternative_suggestions is True:
            for position in range(len(next_samples)):
                df.loc[df.index.isin([next_samples[position]]),"priority"] = (9-position)/10

        priority_list = df["priority"]

        return priority_list
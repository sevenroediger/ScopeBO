import os
import random
import sys
import warnings

from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.samplers import SobolQMCNormalSampler
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling, CVTSampling
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import torch

from .mlr_modeling import regression_modeling
from .model import build_and_optimize_model
from .space_creator import create_search_space
from .utils import EDBOStandardScaler, calculate_vendi_score, obtain_full_covar_matrix, vendi_pruning, variance_pruning, SHAP_analysis, draw_suggestions
from .acquisition import greedy_run, explorative_run, random_run, low_variance_selection, hypervolume_improvement
from .featurization import calculate_morfeus_descriptors
from .visualize import UMAP_view

# torch settings
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu")}


class ScopeBO:
    """
    Main class for the ScopeBO package.
    Contains functions for: 
        - Generating features from SMILES strings (generate_features function)
        - Creating the reaction space from reactant feature files (create_reaction_space function)
        - Analyzing feature importance using SHAP
        - Predicting reaction performance using multivariable linear regression (predict_performance function)
        - Visualizing the reaction space using UMAP (visualize function)
        - Calculating the Vendi score for evaluated samples (get_vendi_score function)
        - Running the ScopeBO optimization loop to suggest experiments (run function)
    """

    def __init__(self):
        self.full_covariance_matrix = None


    @staticmethod
    def generate_features(smiles_list,
                          filename = "reactant_features.csv",
                          common_core=None,
                          chunk_size=10,
                          find_restart = True,
                          starting_smiles_nr=1,
                          chunk_label=1):
        """
        Generates featurization of the reactants using Morfeus descriptor from a SMILES list.
        We recommend to use DFT features, but if these are not available, this function can be
        used to generate featurization without the computational overhead associated with 
        DFT calculations.
        Although the morfeus calculations are much faster than DFT calculations, they can still
        take some time. The feature calculation can be interrupted and continued at the a later
        time by simply running the function again.

        NOTE: The function generates a temporary folder "featurization_temp" during execution
        that is deleted at the end of the run. In case of an interrupted run, this folder serves
        as a storage for the calculated chunks. Do not delete this folder if you want to continue
        an interrupted calculation at a later point!
        --------------------------------------------------------------------------------------
        Inputs:
            smiles_list: list
                list of smiles strings
            common_core: str or None
                SMARTS for the common core of interest
                Default is None --> will look for the largest common substructure in the molecule
            filename: str
                path for the generated dataset
                Default: "reactant_features.csv"
            common_core: str or None
                SMARTS for a substructure for which atom descriptors will be extracted
                If common_core=None (Default), the atom descriptors will be calculated for the
                maximum common substructure.
            chunk_size: int
                number of compounds that will be calculated in one chunk before saving the obtained data
                At the end of the run, all chunks will be concatenated.
                Default: 10
            find_restart: Boolean
                If True, the  algorithm will parse if some chunks were already calculated and auto-restart 
                with the next chunk, overwriting the starting_smiles_nr and chunk_label variables.
            starting_smiles_nr: int (one-indexed)
                first entry of the smiles list to be calculated
                Default: 1
                (useful for restarting in case the calculation crashes)
                NOTE: overwritten if find_restart = True
            chunk_label: int (one-indexed)
                label for the next chunk to be calculated
                Default: 1
                (useful for restarting in case the calculation crashes)
                NOTE: overwritten if find_restart = True
        --------------------------------------------------------------------------------------
        Returns:
            Generates a csv file for all reactants with their featurization.
            Returns the featurization data as a dataframe.
        """

        # Call the function from featurization.py
        _, df_combined = calculate_morfeus_descriptors (smiles_list = smiles_list, filename = filename,
                                       common_core = common_core, chunk_size = chunk_size,
                                       find_restart = find_restart,
                                       starting_smiles_nr = starting_smiles_nr,
                                       chunk_label = chunk_label)
        
        return df_combined
    
    
    @staticmethod
    def create_reaction_space(reactants,
                              feature_processing=True,
                              suggest_samples=True,
                              objectives=None,
                              draw_suggested_samples=True,
                              directory='./',
                              filename='reaction_space.csv'):
        """
        Creates a reaction space csv file with all possible scope combinations and directly suggests three 
        initial scope entries by random sampling.
        ------------------------------------------------------------------------
        
        reactants: list or dictionary
            list of csv file names (as strings). One file per starting material.
                Example: ['reactant1.csv','reactant2.csv']
            The algorithm will set a prefix for the features associated with each
            reactant. By supplying a dictionary for reactants, the feature prefixes
            can be set (as the dict values). Otherwise, generic "reactant#" prefixes
            will be used.
            The csv files should contain the names of the compounds in the first
            column and the featurization of the compounds in the remaining columns.
            The featurization needs to be nummerical.
                Example:    name    feature1     feature2
                            A       23.1         54
                            B       5.7          80
        suggest_samples: Boolean
            Option to suggest three initial samples after creating the reaction space. Default is True.
        objectives: list or None
            list of the objectives. E. g.: ["yield","ee"]
            If None, an objective column with the name 'yield' will be added by default.
        draw_suggested_samples: Boolean
            Option to draw the suggested initial samples. Default is True.
        feature_processing: Boolean
            Option to preprocess the features. Default is True.
        directory: string
            set the working directory. Default is current directory.
        filename: string
            Filename of the output search space csv file. Default is reaction_space.csv
        ------------------------------------------------------------------------        
        Returns a dataframe df with all search space reaction combinations and prints the suggested random samples.
        """

        # create the search space. The function create_reaction_scope is in space_generator.py.
        # only save the data if suggest_samples is False to avoid double saving
        save_data = True
        if suggest_samples:
            save_data = False
        df = create_search_space(reactants=reactants, feature_processing=feature_processing, save_data= save_data,
                                   directory=directory, filename=filename)
        
        # suggest initial samples if requested (random sampling)
        if suggest_samples:
            print("\nSuggesting initial scope entries by random sampling...")
            if objectives is None:
                msg = "No objective name was provided. An objective column with the name 'yield' will be added by default."
                objectives = ['yield']
                print(msg)

            # suggesting random samples
            wdir = Path(directory)
            csv_filename = wdir.joinpath(filename)
            df = ScopeBO()._init_sampling(df=df, batch=3, seed=42, sampling_method='random')
            # sort by priority, add objectives, and  save
            df.sort_values('priority', ascending=False, inplace=True)
            print("Suggested samples are indicated by priority = 1.\n")
            for objective in objectives:
                df[objective] = ['PENDING'] * len(df)
            # reorder columns to have the priority column at the end
            df = df[[col for col in df.columns if col != 'priority'] + ['priority']]
            df.to_csv(csv_filename, index=True, header=True)

            if draw_suggested_samples:  # print the suggested samples if requested
                draw_suggestions(df=df)      
        # return the searchspace df
        return df


    @staticmethod
    def feature_analysis(filename="reaction_space.csv", 
                         objectives =None,
                         objective_mode = {"all_obj":"max"},
                         plot_type=["bar"],
                         directory="."):
        """
        Analyzes the importance of features on the surrogate model using SHAP.
        ---------------------------------------------------------------------
        Inputs:
            filename: str
                filename of the reaction space csv file including experimental outcomes
            objectives: list
                list of the objectives. E. g.: [yield,ee]
                If None, they are automatically inferred from columns containing
                "PENDING" strings as values.
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
        Returns the shap.explainer object, mean absolute SHAP values, and the requested plot of SHAP values.
        """

        # Call the function from the utils.py file
        shap_values, mean_abs_shap_values = SHAP_analysis(
            objectives=objectives,
            objective_mode=objective_mode,
            filename=filename,
            plot_type=plot_type,
            directory=directory)

        return shap_values, mean_abs_shap_values
    

    @staticmethod
    def predict_performance(filename, objective, 
                        further_objectives = None, n_feat = 3,
                        repeats_outer = 5, k_outer = 4, 
                        repeats_inner = 10, k_inner = 5, 
                        fname_shap = "df_shap.csv",
                        feature_cutoff = 20, corr_cutoff = 0.7,
                        fname_pred = "mlr_predictions.csv",
                        print_pred = True,
                        directory = "."):
        """
        Trains a multivariable linear regression model using a repeated, nested CV scheme 
        based on the scope samples and predicts the performance for the rest of the search 
        space.
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

        # call the outsourced function (in mlr_modeling.py)
        df_pred = regression_modeling(filename, objective, further_objectives, n_feat,
                        repeats_outer, k_outer, repeats_inner, k_inner, fname_shap,
                        feature_cutoff, corr_cutoff, fname_pred, print_pred, directory)
        
        return df_pred

    @staticmethod
    def visualize(filename,
                obj_to_show = None,
                obj_bounds = None,
                objectives = None,
                display_cut_samples = True,
                display_suggestions = True,
                display_alternatives = True,
                figsize = (10,8),
                dpi = 600,
                draw_structures = True,
                show_figure = True,
                cbar_title = None,
                return_dfs = False,
                directory = "."):
        """
        Creates a UMAP for the search space, highlighting the picked samples.
        ----------
        filename : str or Path
            Path to the CSV file containing the reaction search space.
        obj_to_show : str or None
            Name of the objective that is visualized.
            If None (Default), the first listed objective is used.
        obj_bounds : tuple or list, optional
            (max, min) values to manually set the colorbar range for `obj_to_show`.
            If None, the min/max are taken from the observed evaluated samples.
        objectives : list-like, optional
            List of column names containing objective values (including "PENDING").
            If None, they are automatically inferred from columns containing
            "PENDING" strings.
        display_cut_samples : bool, default=True
            Whether cut samples (priority = -1) are shown as X markers.
            If False, they are plotted as unseen points.
        display_suggestions: bool, default=True
            Whether suggested samples (priority=1) are shown as squares.
            If Ffalse, they are plotted as unseen points.
        display_alternatives: bool, default=True
            Whether alternatively suggested samples (0<priority<1) are shown as diamonds.
            Decreasing size indicates decreasing priority.
            If False, they are plotted as unseen points.
        figsize : tuple, default=(10, 8)
            Size of the generated UMAP figure in inches.
        dpi : int, default=600
            Resolution of the output figure.
        draw_structures : bool, default=True
            Draw the structures of the evaluated samples. Requires SMILES strings as index in the CSV file.
        show_figure : bool, default=True
            Whether to display the UMAP plot.
        cbar_title : str, optional
            Custom title for the colorbar. If None, uses the objective name.
        return_dfs : bool, default=False
            If True, returns a dictionary of DataFrames for:
                - seen     (evaluated samples)
                - neutral  (unseen priority = 0)
                - cut      (unseen priority = -1)
        directory : str or Path, default="."
            Directory containing the CSV file.
        """

        # Call the function from visualize.py
        df_dict = UMAP_view(filename=filename, obj_to_show=obj_to_show, obj_bounds=obj_bounds,
                            objectives=objectives, display_cut_samples=display_cut_samples,
                            display_suggestions=display_suggestions, display_alternatives=display_alternatives,
                            figsize=figsize, dpi=dpi, show_figure=show_figure, cbar_title=cbar_title,
                            return_dfs=return_dfs, directory=directory, draw_structures=draw_structures)

        if return_dfs:
            return df_dict

    
    def get_vendi_score(self, objectives = None, directory='.', filename='reaction_space.csv'):

        """
        Calculates the Vendi score for all samples that have been evaluated so far (= training data).
        To do this, the covariance matrix of the surrogate model prior is used to calculate the vendi score.
        ------------------------------------------------------------------------------
        Input:
            objectives: list
                list containing the objective names as string values
                If None, they are automatically inferred from columns containing
                "PENDING" strings as values.
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

        # identify the objectives (containing PENDING entries) if none are given
        if objectives is None:
            objectives = df.columns[df.eq("PENDING").any()].to_list()

        # Sort the df by index to ensure compatibility with the covariance matrix values.
        sorted_df = df.sort_index()

        #get the indices of all datapoints that were evaluated so far. 
        # Samples that have not been measured will have "PENDING" as the entry in the objective column and will be ignored.
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
                Reaction space from create_reaction_scope
            batch: float
                Number of experiments to suggest.
            seed: int
                random seed for reproducibility
            sampling_method: String
                Selected sampling method.
                Options:    random --> default
                            lhs (LatinHypercube)
                            cvt (CVTSampling)
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
                # Add the additional samples to the samples dataframe. 
                # If some of the additional_samples are already in samples, generate new ones until the batch size is reached.
                extra_seed = 45
                while len(samples) < batch:
                    samples = pd.concat([samples,additional_samples]).drop_duplicates(ignore_index=True)
                    additional_samples = df.sample(n=batch-len(samples), random_state=seed+extra_seed, replace=True)
                    extra_seed += extra_seed
                
        # Samples have been created, but need to be assigned a priority and also to the dataframe.

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
            objectives, 
            objective_mode = {"all_obj":"max"}, 
            objective_weights=None,
            directory='.', 
            filename='reaction_space.csv',
            batch=3, 
            init_sampling_method='random', 
            seed=42,
            Vendi_pruning_fraction=13,
            pruning_metric = "vendi_batch",
            acquisition_function_mode='balanced',
            give_alternative_suggestions=True,
            show_suggestions=True,
            sample_threshold=None,
            enforce_dissimilarity=False
            ):
        
        """
        ScopeBO main function to suggest experiments based on previous experimental results.
        Returns the updated reaction space dataframe with suggested experiments having the highest priority.
        The input search space csv file (variable filename) is overwritten with the new priority values.
        priority values:
            1 : suggested samples 
            0.9–0.5: alternative suggestions (only if give_alternative_suggestions = True) (higher value = higher priority)
            0  : unseen samples (not yet suggested for experimentation)
            -1 : cut samples (not suggested for experimentation)
            -2 : previously evaluated samples
        Also visualizes the suggested experiments if show_suggestions = True. 
        The visualization however only works if the search space indices are SMILES. Use show_suggestions = False otherwise.
        ------------------------------------------------------------------------
        objectives: list
            list of strings containing the name for each objective.
            Example:
                objectives = ['yield', 'selectivity', 'purity']
        objective_mode: dict
            Dictionary of objective modes for objectives
            Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
            Code will assume maximization for all non-listed objectives
            Default is {"all_obj":"max"} --> all objectives are maximized
        objective_weights: list or None
            list of float weights for the scalarization of the objectives 
            only relevant for multi-objective greedy runs, not other acquisition functions
            Default: None (objectives will be averaged)
        directory: string
            name of the directory to save the results of the optimization.
            Default is the current directory (".")
        filename: string
            Name of the search space file containing the possible scope selections and prior results.
            Default name is 'reaction_space.csv'.
        batch: int
            Number of experiments that you want to run in parallel. 
            Default is 3 (optimized settings).
        init_sampling_method: string:
            Sampling method for selecting the first samples in the scope. Choices are:
            - 'random' : Random sampling (default).
            - 'lhs' : LatinHypercube sampling.
            - 'cvt' : CVT sampling (default option) 
        seed: int
            Seed for reproducibility. Default = 42
        Vendi_pruning_fraction: int
            Pruning percentage for removal of similar samples from the search space.
            Default is 13 (optimized settings).
        pruning_metric: str
            Mode used for the pruning.
            Options:
                "vendi_batch": pruning by vendi scores before every round of experiments (default - optimized settings).
                "vendi_sample": pruning by vendi score before every sample.
                "variance": pruning by surrogate model variance (implemented only for benchmarking purposes).
        acquisition_function_mode: str
            Choose the acqusition function.
            Options:
                "balanced" (Default): exploration-exploitation trade-off via qExpectedImprovement (1 objective) 
                                                                          or qNoisyExpectedHypervolumeImprovement (multi-objective)
                "greedy": pure exploitative selection
                "explorative": pure explorative selection
                "random": random selection
        give_alternative_suggestions: Boolean
            Option to get 5 alternative suggestions. These can be used if the preferred suggestion is not feasible experimentally.
            Default is True.
        show_suggestions: Boolean
            Option to draw the suggested experiments after the run. Only works if the search space indices (compound identifiers) are SMILES.
            Default is True.
        sample_threshold: float, tuple, or None
            Numeric threshold for a minimum pairwise Vendi score between two samples in a batch.
            If a tuple is provided, the first value is the Vendi score threshold 
            and the second value is the minimum number of prior samples needed for the pruning to be applied.
            Default is None (no thresholding).
            This option was explored during development, but not used in the final version of ScopeBO 
            and is only implemented for legacy reasons.
        enforce_dissimilarity: Boolean
            If True, removes all samples from the search space in each batch that have a pairwise 
            Vendi score below 1.06 to any of the previously selected samples.
            Default is False.
            This option was explored during development, but not used in the final version of ScopeBO 
            and is only implemented for legacy reasons.
        """
        
        # Set filenames, random seeds.
        wdir = Path(directory)
        csv_filename = wdir.joinpath(filename)
        torch.manual_seed(seed)
        np.random.seed(seed)
  

        # Check for correct input.

        # Check for correct Vendi_pruning_fraction input.
        if Vendi_pruning_fraction is not None:
            msg = "Vendi_pruning_fraction must be between 0 (no pruning) and 100 (all samples pruned) if provided. Please check your input."
            assert (Vendi_pruning_fraction >= 0 and Vendi_pruning_fraction <= 100), msg

        # Check if objectives is a list (even for single objective optimization).
        if type(objectives) != list:
            objectives = [objectives]

        # Assert that the correct number of weights are given if they are provided
        if objective_weights is not None:
            msg = "The number of objective weights does not match the number of objectives. Please check your input."
            assert (len(objective_weights) == len(objectives)),msg
            # make sure the weights are all floats
            objective_weights = [float(weight) for weight in objective_weights]

        # Check that the reaction space table exists.
        msg = "The reaction space file was not found. Please create one and provide it as input (csv file)."
        assert os.path.exists(csv_filename), msg

        # Load reaction space from scope csv file and remove columns without any values.
        df = pd.read_csv(f"{csv_filename}",index_col=0,header=0, float_precision = "round_trip")
        df = df.dropna(axis='columns', how='all')
        original_df = df.copy(deep=True)  # Make a copy of the original data.

        # get the objectives that are actually in the scope DataFrame.
        obj_in_df = list(filter(lambda x: x in df.columns.values, objectives))

        # Check whether new objective has been added
        # if there are, add them to the DataFrame and use PENDING as a dummy value.
        for obj_i in objectives:
            if obj_i not in original_df.columns.values:
                original_df[obj_i] = 'PENDING'

        # check if there are DataFrame entries with experimental results (no PENDING values in any objective column)
        idx_experimental_results = original_df[~original_df.isin(['PENDING']).any(axis=1)].index

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

        # If there are no experimental results, use initialization sampling.
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
            
        # There are experimental results that can be used to train the model.
        print(f"Found {len(idx_experimental_results)} existing scope entries.")
        
        # Check that the search space still contains enough samples. Also reset priority of suggested samples that were not measured.
        if "priority" in df.columns.values:
            # test samples are samples without experimental results (= rows containing "PENDING") and which were not pruned (priority != -1)
            df_noexperiments = df[df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]
            idx_test = df_noexperiments[df_noexperiments['priority'] != -1].index
            
            # Check if there are less samples in the search space than the batch size.
            if len(idx_test) < batch:
                if len(idx_test) == 0:
                    print("There no more samples left in the search space.")
                    # The latest samples have priority 1 - change it to -2 to indicate that they have been run
                    df_experiments = df[~df.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]
                    idx_train = df_experiments.index
                    for idx in idx_train:
                        df["priority"].at[idx] = -2
                    return original_df
                else:
                    batch = len(idx_test)
                    if batch == 1:
                        print(f"These is only 1 sample left in the search space. The batch size is thus decreased to 1.")
                    else:
                        print(f"There are only {batch} samples left in the search space.")
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
        Runs the BO process using Vendi pruning, a Gaussian Process surrogate model 
        and expected improvement-type acquisition functions:
            qExpectedImprovement for single objective models
            qNoisyExpectedHypervolumeImprovement for multi objective models
        (These are the default acquisition functions, but also others can be requested via the acquisition_function_mode parameter).

        Returns a priority list for a given reaction space (top priority to low priority).
        -------------------------------------------------------        
        df:   Dataframe for the BO run
                (both test+train, see run function)
        
        batch: int (number of experiments to run in this batch of experiments)

        full_covariance_matrix: DataFrame
            covariance matrix of the full dataset
        
        Other variables:
            see doc string for run function above.
        """

        class HiddenPrints:
            """Class to hide print statements from functions called within the function."""
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
        # if requested (default is False)
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

        # prepare X and y data for BO model by removing objectives and priority columns for the BO model inputs
        df_train_y = df.loc[idx_train][objectives]
        if 'priority' in df.columns.tolist():
            priority_list = list(df["priority"])
            df = df.drop(columns=objectives + ['priority'])
        else:
            df = df.drop(columns=objectives)
        df_train_x = df.loc[idx_train]
        df_test_x = df.loc[idx_test]
        
        # Check the number of objectives.
        n_objectives = len(df_train_y.columns.values)

        # Scaling of input data (normalization).
        scaler_x = MinMaxScaler()
        scaler_x.fit(df_train_x.to_numpy())  # fit on training features
        train_x_np = scaler_x.transform(df_train_x.to_numpy())  # transform training features
        test_x_np = scaler_x.transform(df_test_x.to_numpy())  # transform test features with the same scaler
        
        # Scaling of training outputs (standardization). 
        # Also convert minimization problems to pseudo-maximization problem by negating the output values.
        train_y_np = df_train_y.astype(float).to_numpy()
        min_obj = [obj for obj, value in objective_mode.items() if value == "min"]
        if min_obj:
            for obj in min_obj:
                i = objectives.index(obj)
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

        # Prune the search space via Vendi scoring if requested (this pruning mode is the default)
        if Vendi_pruning_fraction != 0:
            if pruning_metric.lower() == "vendi_batch":
                cumulative_test_x, cut_by_vendi, idx_test = vendi_pruning(
                    idx_test = idx_test,
                    idx_train = idx_train, 
                    Vendi_pruning_fraction = Vendi_pruning_fraction,
                    cumulative_test_x = cumulative_test_x, 
                    cut_by_vendi = cut_by_vendi,
                    full_covariance_matrix=full_covariance_matrix,
                    df=df,
                    seed = seed)
                print(f"Cut {len(cut_by_vendi)} samples from the search space via Vendi pruning. {len(idx_test)} samples remain.")
                
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
                print(f"Selecting sample {batch_exp+1} of {batch}...")

                # Prune the search space via Vendi scoring if requested 
                # (this time for each sample (after the fantasies) not each batch).
                # Default is pruning before each batch (see above), not before each sample (here)
                if ((pruning_metric.lower()  == 'vendi_sample') and (Vendi_pruning_fraction != 0)):
                    cumulative_test_x, cut_by_vendi, idx_test = vendi_pruning(
                        idx_test = idx_test,
                        idx_train = idx_train, 
                        Vendi_pruning_fraction = Vendi_pruning_fraction,
                        cumulative_test_x = cumulative_test_x, 
                        cut_by_vendi = cut_by_vendi,
                        full_covariance_matrix=full_covariance_matrix,
                        df=df,
                        seed = seed)
                    print(f"Cut {len(cut_by_vendi)} samples from the search space via Vendi pruning. {len(idx_test)} samples remain.")

                # Instantiate some variables.
                surrogate_model = None
                train_x_torch = None
                test_x_torch = None

                # surrogate modeling unless random acquisition function is selected
                if acquisition_function_mode.lower() != "random":   
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
                        surrogate_model = individual_models[0]  # model is directly the SingleTaskGP object
            
                # Instantiate some variables for the acquisition function evaluation.
                acquisition_function = None 
                sample = None
                samples = None
                idx_sample = None
                list_position_sample = None

                # Balanced acquisition function (exploration-exploitation trade-off -- default behavior).
                if acquisition_function_mode.lower() == "balanced":
                    if n_objectives > 1:  # Acquisition function for multi-objective optimization (qNoisyEHVI)

                        # Reference point is the minimum seen so far (important for hypervolume calculation).
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

                        # Get the sample index and save it.
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
                        surrogate_model=surrogate_model, q=1, objective_weights=objective_weights,
                        idx_test=idx_test, test_x_torch=test_x_torch)
                    
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

                # Exploitative acquisition function. Implemented for benchmarking.
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
                    if n_objectives == 1:  # single objective greedy run
                        idx_sample, sample, list_position_sample = greedy_run(
                            surrogate_model=surrogate_model, q=acquisition_samples, objective_weights=objective_weights,
                            idx_test=idx_test, test_x_torch=test_x_torch)
                    else:  # multi-objective greedy run
                        idx_sample, sample, list_position_sample = hypervolume_improvement(
                            surrogate_model=surrogate_model, q=acquisition_samples, objective_weights=objective_weights,
                            cumulative_train_y=cumulative_train_y,idx_test=idx_test, test_x_torch=test_x_torch)
                        
                    # Update batch fantasy.
                    idx_test = np.delete(idx_test, list_position_sample[0])
                    idx_train = np.append(idx_train, idx_sample[0])
                    popped_sample = cumulative_test_x.pop(list_position_sample[0])
                    cumulative_train_x.append(popped_sample)
                    best_samples.append(idx_sample[0])
                    if len(idx_sample) > 1:
                        for sample in idx_sample[1:]:  # the first entry of the list is the actual suggestion, the rest are the alternatives
                            next_samples.append(sample)
   
                # Update the fantasy with the predicted value for the selected sample.
                if (acquisition_function_mode.lower() != "random") and (batch_exp != batch -1):
                    # Get the yield prediction for the suggested sample and save it for the fantasy.
                    y_pred = surrogate_model.posterior(
                        torch.tensor(sample)).mean.detach().numpy()[0].tolist()
                    cumulative_train_y.append(y_pred)              

        # selection based on low variance score without batch fantasies (implemented for benchmarking purposes, only single objective possible)
        elif acquisition_function_mode.lower() == "low_variance":
            best_samples = low_variance_selection(batch, idx_test, cumulative_test_x, cumulative_train_x, cumulative_train_y)

        else:
            return print("Acquisition function not found - no samples selected. Please check your input!")

        # remove samples that are too similar to each other in the suggestions if requested (default is to not do this)
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
                print("The input variable sample_threshold must either be numeric or a tuple. Please check your input!")
            
            if (sample_cutoff is None) or (sample_cutoff >= (len(idx_train))-3):  # minus 3 because this run's samples have already been added to idx_train

                # calculate the pairwise Vendi scores of the pruned samples and remove those that are too similar to each other
                # generate pairs out of the suggested samples
                sorted_df = df.sort_index()
                pairs = list(itertools.combinations(best_samples, 2))
                dict_pairwise  = [{'sample1': pair[0], 'sample2': pair[1], 
                                'Vendi score': np.nan, 'idx_num1': sorted_df.index.get_loc(pair[0]), 
                                'idx_num2': sorted_df.index.get_loc(pair[1])} for pair in pairs]
                df_pairwise = pd.DataFrame(dict_pairwise)
                
                # calculate the pairwise Vendi scores
                df_pairwise["Vendi score"] = [calculate_vendi_score(idx_num=[df_pairwise.loc[round,"idx_num1"],
                                                                             df_pairwise.loc[round,"idx_num2"]],
                                                                    covariance_matrix=self.full_covariance_matrix) for round in df_pairwise.index]

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
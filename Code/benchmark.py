import ast
import colorsys
import os
from pathlib import Path
import re
from statistics import stdev
import sys

from IPython.display import display
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import seaborn as sns
from adjustText import adjust_text

from .predictor  import ScopeBO
from .utils import calculate_vendi_score, obtain_full_covar_matrix

# General plt parameters
plt.rcParams.update({
    "axes.titlesize": 20,        # Subplot title
    "axes.labelsize": 16,        # X and Y labels
    "figure.titlesize": 24,      # Suptitle
    "xtick.labelsize": 14,       # X tick labels
    "ytick.labelsize": 14,       # Y tick labels
    "legend.fontsize": 14,       # Legend text
    "legend.title_fontsize": 14, # Legend titles
    "font.family": "Helvetica"   # Font
    })


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Benchmark:
    """
    Class for running full scopes, collecting their data, and analyzing the results.
    
    Functions for data collection:
        collect_data: 
            main functionality for running scopes
        change_featurization:
            recalculating Vendi scores in existing benchmark data with a different featurization
            (essentially creates a new scope run folde with these updated values)

    Functions for data analysis:
        heatmap plot:
            plot the overall results of scopes
        progress_plot:
            plot the progression of scopes with increasing scope size
        track_samples:
            visualize the selected samples on a UMAP
        show_scope:
            draw the structures of the selected compounds
        feature_analysis:
            SHAP analysis of the surrogate model for a scope
        objective_distribution:
            analyze the distribution of objective values in a scope
        get_metric_overview:
            collect the different metrics for all scopes in a benchmark folder (all random seeds)


    Utility functions:
        normalization:
            normalize values
        standardization:
            standardize values
        calculate_scope_score:
            calculate scope scores
        find_objectives:
            look up the objectives that were used in a scope
        _adjust_lightness:
            adjust the lightness of a color. Used for colormap definition.

    """

    def __init__(self):
        
        # Define colormaps for plotting
        doyle_colors = ["#CE4C6F", "#1561C2", "#188F9D","#C4ADA2","#515798", "#CB7D85", "#A9A9A9"]
        
        # extension of palette with lighter and darker versions
        lighter = [self._adjust_lightness(c, 1.2) for c in doyle_colors]
        darker  = [self._adjust_lightness(c, 0.7) for c in doyle_colors]
        self.all_colors = doyle_colors + darker[::-1] + lighter[::-1] 

        # Save the categorical colormap
        self.cat_cmap = ListedColormap(self.all_colors, name="Doyle_cat")

        # Define and save a continuous colormap
        colors = [doyle_colors[1],"#FFFFFFD1",doyle_colors[0]]
        self.cont_cmap = LinearSegmentedColormap.from_list("Doyle_cont", colors)

        # Define and save a second continuous colormap
        colors = [doyle_colors[3],"#FFFFFFD1",doyle_colors[2]]
        self.cont_cmap2 = LinearSegmentedColormap.from_list("Doyle_cont2", colors)

    
    @staticmethod
    def _adjust_lightness(color, factor=1.2):
        """
        Helper function to make colors lighter (factor > 1) or darker (factor < 1).
        Used for colormap definition.
        """
        r, g, b = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, min(1, l * factor))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return mcolors.to_hex((r, g, b))

    
    def collect_data(self, filename_labelled, objectives, name_results, specific_seed = None, seeds = 40, idx_to_keep = None,
                     objective_mode = {"all_obj":"max"}, init_sampling_method="random", Vendi_pruning_fractions = [13], batches=[3], budget=27, 
            objective_weights = None, sample_threshold = None, enforce_dissimilarity=False, pruning_metric = "vendi_batch", 
            acquisition_function_mode = 'balanced', dft_filename = None, filename_prediction = "df_benchmark.csv", directory='.'):
        """
        Runs the ScopeBO.run function for selected parameter ranges and records the results per round for each set of parameters.

        Returns a number of csv files in a folder with the name name_results:
            Raw results:
                One per combination of seed, batch, and Vendi_pruning_fraction settings containing the following results:
                objective values for each evaluated point, grouped by round ("obj_value")
                Vendi score of the all evaluated points after each round ("Vendi_score")
                indices of evaluated samples for each round ("eval_samples")
                indices of samples cut by the Vendi cutoff for each round ("cut_samples")

                Example output (batch size 3 and 2 objectives; 5 samples were pruned from the reaction space):

                        objective_values        Vendi_score     samples     cut_samples
                0       [[1,2,5],[1,5,12]]     43              [3,5,12]    [5,3,1,35,5]
                1       [[11,2,11],[16,1,4]]   37              [5,17,1]    [4,9,7,24,51]

                Whereas the index column corresponds to the round of experiments.

                The raw result files are saved as [budget][acquisition_function_mode]_b[batch]_V[Vendi_pruning_fraction]_s[seed].csv
                They are located in a subfolder "raw_data".

                If specific_seed != None, the raw results are directly located in name_results.

            Summarized data:
                These files contain either objective value data (averaged for each batch round) ("obj") or
                Vendi score data ("vendi"). The rows correspond to the different batch sizes in batches (noted as indices). The columns
                correspond to the different Vendi_pruning_fractions.
                The entries are lists containing the results per round. There are files contain average values ("average") (across the different seeds) or 
                the standard deviation ("stdev") (across the different seeds).
                In the case of multiple objectives, a separate file for each objective value as well as a file containing the combined objective values are generated.

                Example name for a run with the objectives "yield" and "selectivity": "benchmark_obj[yield__selectivity]_av.csv" --> average objective value data
                
                Summarized data is only generated if the variable specific_seed = None (default).
        
        -------------------------------------------------------------------------------------

        filename_labelled: string
            name of the csv file with the labelled data
        objective: list
            list indicating the column name for the objective (string)
        name_results: str
            name of the folder in which the results will be saved
        specific_seed: int or None
            Ignored if None (default).
            If specified, will only run the given random seed (overwriting variable seeds).
        seeds: int
            number of different random seeds to be evaluate (will use range(seeds))
            Default is 40.
        idx_to_keep: list or None
            list of indices of the labelled dataframe which won't be removed at the beginning of the scope run
            Enables to "hot-start" a scope with prior data.
            Default is None (remove all idx and start from the scope from scratch).
        objective_mode: dict
            Dictionary of objective modes for objectives
            Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
            Code will assume maximization for all non-listed objectives
            Default is {"all_obj":"max"} --> all objectives are maximized   
        init_sampling_method: str
            sampling method. Options:
                "random": random selection (default)
                "cvt": CVT sampling
                "lhs": LHS sampling
        Vendi_pruning_fractions: list
            list of threshold values for the Vendi cutoff (ints)
            Default: [13]  
        batches: list
            list of batch sizes (as int numbers)
            Default: [3]
        budget: int
            experimental budget
            Default: 27
        objective_weights: list or None
            list of float weights for the scalarization of the objectives 
            only relevant for multi-objective greedy runs, not other acquisition functions
            Default: None (objectives will be averaged)
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
        dft_filename: None or str
            name of the file containing the dft-featurized reaction space for the vendi score calculation
            only has to be given if the substrate encoding is not dft-derived (e. g. Mordred or Rdkit featurization)
            NOTE: only implemented for mono-objective cases
            Default: None
        filename_prediction: str
            name of the csv file in which the benchmark dataframe is saved
            Default: "df_benchmark.csv"
        directory: str
            working directory (default is current directory)
        """
        
        # Create the results folder and the folder for raw results.
        wdir = Path(directory)
        if not os.path.exists(wdir.joinpath(name_results)):
            # Create the folder
            os.makedirs(wdir.joinpath(name_results))
        if (not os.path.exists(wdir.joinpath(name_results+"/raw_data"))) and (specific_seed is None):
            # Create the folder
            os.makedirs(wdir.joinpath(name_results+"/raw_data"))

        # Read labelled data.
        df_labelled = pd.read_csv(wdir.joinpath(filename_labelled),index_col=0,header=0, float_precision = "round_trip")

        # Generate a copy of the DataFrame with labelled data and remove the objective data.
        df_unlabelled = df_labelled.copy(deep=True)
        # check if some samples should be retained for the start of the scope
        if idx_to_keep is not None:
            # delete all objective data apart from the specified ones
            for obj in objectives:
                df_unlabelled.loc[~df_unlabelled.index.isin(idx_to_keep),obj] = "PENDING"
                print("The scope will be hot-started with the following samples:")
                print(f"{df_unlabelled.index[~df_unlabelled.isin(['PENDING']).any(axis=1)].tolist()}") 
        else:
            df_unlabelled.drop(columns=objectives,inplace=True)

        # Set up the Vendi pruning fraction and batch size names for the results dataframes.
        Vendi_names = []
        for Vpf in Vendi_pruning_fractions:
            if type(Vpf) is list:
                rounded_Vpf = [round(el,1) if type(el) is float else el for el in Vpf]
                Vendi_names.append("-".join(map(str,rounded_Vpf)))
            else:
                Vendi_names.append(str(Vpf))
        batch_names = []
        for batch in batches:
            if type(batch) is list:
                batch_names.append("-".join(map(str,batch)))
            else:
                batch_names.append(str(batch))

        # Instantiate empty df for the analyzed results.
        df_obj_av = pd.DataFrame(None,batch_names,Vendi_names)
        df_obj_stdev = pd.DataFrame(None,batch_names,Vendi_names)
        dfs_indiv_obj_av = None
        dfs_indiv_obj_stdev = None
        if len(objectives) > 1:  # also make separate dfs for the individual objectives if there are multiple objectives
            dfs_indiv_obj_av = {}
            dfs_indiv_obj_stdev = {}
            for objective in objectives:
                dfs_indiv_obj_av[objective] = pd.DataFrame(None,batch_names,Vendi_names)
                dfs_indiv_obj_stdev[objective] = pd.DataFrame(None,batch_names,Vendi_names)
        df_vendi_av = pd.DataFrame(None,batch_names,Vendi_names)
        df_vendi_stdev = pd.DataFrame(None,batch_names,Vendi_names)

        # Instantiate a ScopeBO object
        myScopeBO = ScopeBO()

        # Set the name for the file used in the benchmarking runs.
        csv_filename_pred = wdir.joinpath(filename_prediction)

        # In case of non-DFT featurization, set up some other things to get a DFT-featurization-based vendi score.
        vendiScopeBO = None
        df_dft = None
        filename_vendi = None
        current_df_dft = None
        if dft_filename:
            filename_vendi = filename_prediction.replace(".csv","_vendi.csv")
            df_dft = pd.read_csv(wdir.joinpath(dft_filename),index_col=0,header=0, float_precision = "round_trip")
            df_dft[objectives[0]] = "PENDING"
            vendiScopeBO = ScopeBO()
            
        # Run all the requested parameter settings.
        run_counter = 1  # variable for feedback during run

        # test the number of random seeds as indicated in the variable seeds
        seeds_to_test = range(seeds)
        # if a specific seed was requested, only test that one
        if specific_seed is not None:
            seeds_to_test = [specific_seed]

        total_runs = len(batches) * len(Vendi_pruning_fractions) * len(seeds_to_test)

        # Loop through all combinations of batch sizes, Vendi pruning fractions, and random seeds.
        for batch in batches: 
            for Vpf in Vendi_pruning_fractions:
                
                seeded_list_obj = []
                seeded_list_vendi = []

                for seed in seeds_to_test:     
                    # Reset the csv file for the campaign by removing the objective data (meaning overwriting with the unlabelled df).
                    df_unlabelled.to_csv(csv_filename_pred, index=True, header=True)
                    if df_dft is not None:
                        current_df_dft = df_dft.copy(deep=True)  # reset the dft-feautrized df for the vendi calculation
                    
                    # Set up lists to hold raw results and average results for this run.
                    raw_results = []
                    run_results = []
                    
                    # Determine the number of rounds of experiments for the given batch size.
                    rounds = 0
                    if type(batch) is list:
                        rounds = len(batch)
                    else:
                        if budget % batch != 0:
                            rounds = int(budget/batch)+1  # extra round with reduced batch size for last run (will be reduced below)
                        
                        else:
                            rounds = int(budget/batch)
                    
                    # Run ScopeBO for these settings.
                    for current_round in range(rounds):
                            
                        # check if the batch size is dynamic (meaning different batch sizes for each rounds)
                        current_batch = None
                        if type(batch) is list:
                            current_batch = batch[current_round]
                        else:
                            current_batch = batch
                            # Check if this will be a run with reduced batch size (due to the set budget).
                            if current_round+1 == rounds and budget % batch != 0:
                                current_batch = budget % batch

                        # Check if the Vendi_pruning_fraction is dynamic (meaning different fractions for each round)
                        this_Vendi_pruning_fraction = Vpf
                        if type(Vpf) is list:
                            this_Vendi_pruning_fraction = Vpf[current_round]

                        # assign labels for the print-out
                        batch_label = batch
                        Vpf_label = this_Vendi_pruning_fraction
                        current_batch_label = current_batch

                        print(f"Now running Batch size: {batch_label}, Vendi_pruning_fraction: {Vpf_label}, Seed: {seed}, Round: {current_round}, current batch: {current_batch_label}")
                        with HiddenPrints():
                            current_df = myScopeBO.run(
                                objectives = objectives,
                                objective_mode= objective_mode,
                                objective_weights = objective_weights,
                                filename = filename_prediction,
                                batch = current_batch,
                                init_sampling_method = init_sampling_method,
                                seed = seed,
                                Vendi_pruning_fraction = this_Vendi_pruning_fraction,
                                pruning_metric = pruning_metric,
                                acquisition_function_mode = acquisition_function_mode,
                                give_alternative_suggestions = False,
                                show_suggestions=False,
                                sample_threshold=sample_threshold,
                                enforce_dissimilarity=enforce_dissimilarity
                            )
                    
                        current_raw_results = []

                        # Save indices of samples.
                        current_idx_samples = list(current_df[current_df["priority"]  == 1].index)
                        if current_idx_samples == []:
                            return print("The scope could not be finished because there are no samples left. Please adjust the settings.")

                        # Update dataframe with results and save the objective values.
                        current_obj = []
                        for objective in objectives:
                            obj_list = []
                            for idx in current_idx_samples:
                                current_df.loc[idx,objective] = df_labelled.loc[idx,objective] 
                                obj_list.append(df_labelled.loc[idx,objective])
                            current_obj.append(obj_list)

                        # Save the dataframe for the next round of ScopeBO.
                        current_df.to_csv(csv_filename_pred, index=True, header=True)

                        # Calculate the Vendi score for all points that were obseved so far.
                        if vendiScopeBO:  # this is the scenario when a non-dft featurization is used in the campaign
                            for idx in current_idx_samples:
                                current_df_dft.loc[idx,objectives[0]] = df_labelled.loc[idx,objectives[0]]
                            current_df_dft.to_csv(wdir.joinpath(filename_vendi),index=True,header=True)
                            current_vendi_score = vendiScopeBO.get_vendi_score(objectives = objectives, 
                                                                            directory = directory, filename = filename_vendi)
                            print("Vendi score calculated via additional file with dft featurization.")                              
                        else:  # this is the standard case for a campaign using dft featurization
                            current_vendi_score = myScopeBO.get_vendi_score(objectives = objectives, directory = directory, filename = filename_prediction)

                        # Get the newly pruned samples by looking up all pruned samples and removing the ones that were already pruned.
                        current_idx_cut = list(current_df[current_df["priority"]  == -1].index)
                        for i in range(current_round):  # loop through the previously saved batches.
                            for j in raw_results[i][3]:  # cut samples are saved as the 4th entry in each current_results list
                                current_idx_cut.remove(j)

                        # Save results for this round in a list and append it to the overall results list.
                        current_raw_results.append(current_obj)
                        current_raw_results.append(current_vendi_score)
                        current_raw_results.append(current_idx_samples)
                        current_raw_results.append(current_idx_cut)
                        raw_results.append(current_raw_results)

                        # Average the objective value for all samples in this round (separately for every objective)
                        current_obj_av = [np.mean(sublist) for sublist in current_obj]

                        # Save the processed results for this round.
                        run_results.append([current_obj_av,current_vendi_score])

                    # Save the processed results for this run.
                    seeded_list_obj.append([run_results[i][0] for i in range(len(run_results))])
                    seeded_list_vendi.append([run_results[i][1] for i in range(len(run_results))])
                    
                    # Save raw results as a csv.
                    df_results = pd.DataFrame(raw_results,columns=[f"obj_values {objectives}","Vendi_score","eval_samples","cut_samples"])
                    if specific_seed is None:
                        csv_filename_results = wdir.joinpath(name_results+f"/raw_data/{budget}{acquisition_function_mode}_"\
                                                             f"b{batch_names[batches.index(batch)]}_V"\
                                                                f"{Vendi_names[Vendi_pruning_fractions.index(Vpf)]}_s{seed}.csv")
                    else:  # no raw_dat subfolder if only one specific seed was requested
                        csv_filename_results = wdir.joinpath(name_results+f"/{budget}{acquisition_function_mode}_b"\
                                                             f"{batch_names[batches.index(batch)]}_V"\
                                                                f"{Vendi_names[Vendi_pruning_fractions.index(Vpf)]}_s{seed}.csv")

                    if len(df_results.iloc[0,0]) == 1:  # flatten unnecessary list of lists for mono-objective runs
                        for idx in df_results.index:
                            df_results.loc[idx,f"obj_values {objectives}"] = str(df_results.loc[idx,f"obj_values {objectives}"][0])
                    df_results.to_csv(csv_filename_results,index=True,header=True)

                    print (f"Finished campaign {run_counter} of {total_runs}.")
                    run_counter+= 1
                
                # only do summary analysis if specific seeds = None
                if specific_seed is None:
                    # Calculate the averages and standard deviations across the different seeds and save the results.
                    obj_value = [[sum(matrix[i][j] for matrix in seeded_list_obj) / len(seeded_list_obj) for j in range(len(seeded_list_obj[0][0]))] for i in range(len(seeded_list_obj[0]))]
                    if len(obj_value[0]) == 1:  # mono-objective
                        df_obj_av.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([value for this_round in obj_value for value in this_round])
                    else:  # multi-objective
                        df_obj_av.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str(obj_value)
                        # also populate the dataframes for the individual objectives
                        for i in range(len(objectives)):
                            dfs_indiv_obj_av[objectives[i]].loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([obj[i] for obj in obj_value])
                    # Standard deviation can only be calculated if there are at least 2 values. Set to zero if there is only one.
                    if seeds > 1:
                        obj_std = [[stdev([matrix[i][j] for matrix in seeded_list_obj]) for j in range(len(seeded_list_obj[0][0]))] for i in range(len(seeded_list_obj[0]))]
                        if len(obj_std[0]) == 1:  # mono-objective
                            df_obj_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([value for this_round in obj_std for value in this_round])
                        else:  # multi-objective
                            df_obj_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str(obj_std)
                            # also populate the dataframes for the individual objectives
                            for i in range(len(objectives)):
                                dfs_indiv_obj_stdev[objectives[i]].loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([obj[i] for obj in obj_std])
                        df_vendi_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]]  = str([stdev(i) for i in zip(*seeded_list_vendi)])
                    else:
                        obj_std = [[0 for j in range(len(seeded_list_obj[0][0]))] for i in range(len(seeded_list_obj[0]))]
                        if len(obj_std[0]) == 1:  # mono-objective
                            df_obj_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([value for this_round in obj_std for value in this_round])
                        else:  # multi-objective
                            df_obj_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str(obj_std)
                            # also populate the dataframes for the individual objectives
                            for i in range(len(objectives)):
                                dfs_indiv_obj_stdev[objectives[i]].loc[str(batch),Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([obj[i] for obj in obj_std])
                        df_vendi_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]]  = str([0 for i in zip(*seeded_list_vendi)])  

                    df_vendi_av.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]]  = str([sum(i) / len(i) for i in zip(*seeded_list_vendi)])

        # Save the dataframes with the processed results as csv files (only if no specific seed).
        if specific_seed is None:        
            df_obj_av.to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{'__'.join(objectives)}]_av.csv"),index=True,header=True)
            df_obj_stdev.to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{'__'.join(objectives)}]_stdev.csv"),index=True,header=True)
            df_vendi_av.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index=True,header=True)
            df_vendi_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_stdev.csv"),index=True,header=True)
            if len(objectives) > 1:
                for objective in objectives:
                    dfs_indiv_obj_av[objective].to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{objective}]_av.csv"),index=True,header=True)
                    dfs_indiv_obj_stdev[objective].to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{objective}]_stdev.csv"),index=True,header=True)

        print(f"Data collection finished! Results are saved in the subfolder {name_results}.")


    @staticmethod
    def change_featurization(name_feat, filename_labelled, name_results, new_folder_name = None, cov_mat = None, directory = "."):
        """
        Recalculate Vendi scores for scopes using an alternative featurization.
        Generates a new folder with recalculated raw data files containing updated ``Vendi_score`` columns.

        Parameters
        ----------
        name_feat : str
            Name of the new featurization used in the Vendi score calculation.
        filename_labelled : str
            Path to the CSV file containing the labelled dataset with the new features.
        name_results : str
            Path to the folder containing the original results.
        new_folder_name: str or None
            Path to the folder where the results will be saved.
            If None is given, the folder will be named automatically (see Returns section).
        cov_mat : np.ndarray, optional
            Covariance matrix to be used for the Vendi score calculation.
            If None (default), the covariance matrix will be calculated from the labelled data.
        directory : str
            Working directory.

        Returns
        -------
        None
            Saves recalculated raw data files with updated "Vendi_score" columns in a 
            new folder named "<name_results>_<name_feat>_feat/raw_data" 
            (or as indicated by the variable new_folder_name).
        """

        # create a folder where the results can be saved
        wdir = Path(directory)
        results_path = wdir / name_results
        raw_path = results_path / "raw_data"
        feat_path = results_path.parent / f"{results_path.name}_{name_feat}_feat/raw_data"
        if new_folder_name is not None:
            feat_path = Path(new_folder_name)
        feat_path.mkdir(parents=True, exist_ok=True)  # ensure that the results folder exists


        # read in the featurization that will be used for the Vendi score calculation
        df_labelled = pd.read_csv(wdir / filename_labelled,index_col=0,header=0)

        # sort it by index
        df_labelled.sort_index(inplace=True)

        # find the objectives
        objectives = Benchmark.find_objectives(name_results,directory)

        # calculate the covariance matrix if none was provided
        if cov_mat is None:
            cov_mat = obtain_full_covar_matrix(objectives=objectives,directory=directory,filename=filename_labelled)

        print(f"Recalcuting the Vendi scores for the scopes in the folder '{name_results}'.")
        # go through all the raw files in the original folder
        for file in os.listdir(raw_path):

            # make an unlabelled copy of the dataframe
            df_Vendi = df_labelled.copy(deep=True)
            for objective in objectives:
                df_Vendi[objective] = "PENDING"

            # read in the raw data file
            df_raw = pd.read_csv(raw_path / file, index_col = 0, header = 0)

            # convert the format of the evaluated samples from one string to a list of strings
            df_raw["eval_samples"] = df_raw["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])

            # reset the Vendi scores
            df_raw["Vendi_score"] = np.nan

            Vendi_scores = []
            # loop through the rounds of the run
            for run_round in df_raw.index:

                    # get the evaluated samples
                    samples = df_raw.loc[run_round,"eval_samples"]

                    # record the evaluated samples in the df of the new featurization
                    for sample in samples:
                        sample = str(sample.encode().decode('unicode_escape'))
                        for objective in objectives:
                            df_Vendi.loc[sample,objective] = df_labelled.loc[sample,objective]

                    # sort to assert that df_Vendi has the required order as the cov_mat
                    df_Vendi.sort_index(inplace=True)

                    # calculate the Vendi score and save it
                    idx_target = (df_Vendi[~df_Vendi.apply(lambda r: r.astype(str).str.contains('PENDING', case=False).any(), axis=1)]).index.values
                    idx_num = [df_Vendi.index.get_loc(idx) for idx in idx_target]
                    current_Vendi_score = calculate_vendi_score(idx_num=idx_num,covariance_matrix=cov_mat)
                    Vendi_scores.append(current_Vendi_score)
        
            # overwrite the Vendi scores in the df
            df_raw["Vendi_score"] = Vendi_scores

            # save it in the new folder
            df_raw.to_csv(feat_path / file, index=True,header=True)

            print(f"Recalculated the Vendi scores in the file {file} (using {name_feat} featurization).")


    def heatmap_plot(self, type_results, name_results, budget, scope_method = "product", 
                    objective_mode = {"all_obj":"max"}, objective_weights = None,
                    bounds = {"rate": (2.349,1.035), "vendi": (6.366,1.941)}, 
                    filename = None, show_plot = True, directory = '.'):
        """
        Generates and saves a heatmap plot for the requested result type (if requested) across different batch sizes and Vendi_pruning_fractions.
        Options for displayed results: scope score ("scope"), vendi score ("vendi"), weighted objectives ("objectives", normalized if multiple objectives),
        or individual objectives displayed by their respective name.
        ---------------------------------------------------------------------------------------------------------
        Input:
            type_results: str
                Requested type of result.
                Options:
                    "vendi": Vendi score
                    "objective": average objective values
                    "scope": scope score
                    You can also provide the name of a specific objective for that run and then it will only analyze that one.
            name_results: str
                Name of the subfolder in which the result csv files are saved.
            budget: int
                experimental budget used in the runs
            scope_method: str
                method to calculate the scope score ("average","product" (Default),"geometric_mean")
            objective_mode: dict
                Dictionary of objective modes for objectives
                Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
                Code will assume maximization for all non-listed objectives
                Default is {"all_obj":"max"} --> all objectives are maximized
            objective_weights: dict or None
                Weights for averaging the objective data
                Example: {"yield": 0.3, "ee": 0.7}
                Default is None --> all objectives will be averaged.
            bounds: dict
                dictionary of bounds for the individual metrics (vendi, all objectives)
                Default values are for the ArI dataset.
            filename: str or None
                name for the figure that is created. Default is None --> figure is not saved.
            show_plot: Boolean
                Option to show the heatmap plot (Default is True.)
            directory: str
                current directory. Default: '.'
        """

        wdir = Path(directory)
        
        # Get the overview for the requested data
        dfs_scaled,_ = self.get_metric_overview(bounds, budget, name_results, type_results, scope_method, objective_mode,
                        objective_weights, directory)
        
        means = dfs_scaled["means"]
        
        # dfs_scaled have the different conditions (batch_pruning) as columns and the scope size as indices
        # transform to grid with batch sizes as indices and pruning amounds as columns
        batch_sizes = list(set([int(col.split("_")[0][1:]) for col in means.columns]))
        pruning_amounts = list(set([int(col.split("_")[1][1:]) for col in means.columns]))
        
        df_heatmap = pd.DataFrame(np.nan,index=batch_sizes,columns=pruning_amounts)
        df_heatmap.index = sorted(df_heatmap.index)
        df_heatmap.columns = sorted(df_heatmap.columns)
        for batch in batch_sizes:
            for pruning in pruning_amounts:
                df_heatmap.loc[batch,pruning] = means.loc[budget,f"b{batch}_V{pruning}"]

        if show_plot:
            # Generate and save the heatmap plot if it was requested.
            plt.figure(figsize=(10,3))
            if type_results == "comb_obj":
                type_results = "Combined objective"
            heatmap = sns.heatmap(df_heatmap,annot=True, fmt=".3f", linewidths=1,cmap=self.cont_cmap,
                                  cbar_kws={'label': f"{type_results} score"})
            heatmap.set(xlabel="Vendi pruning fraction in %", ylabel="batch size")
            heatmap.tick_params(length=0)
            plt.show()
            if filename is not None:
                heatmap_figure = heatmap.get_figure()
                heatmap_figure.savefig(wdir.joinpath(filename))
        return df_heatmap
    

    def progress_plot(self, budget, type_results, name_results, scope_method = "product", objective_mode = {"all_obj": "max"},
                        objective_weights = None, bounds = {"rate": (2.349,1.035), "vendi": (6.366,1.941)}, filename_figure = None, 
                        directory = ".", show_plot = True, show_stats = None, specified_batch_size = None):
            """
            Generates a result(number of experiments) y(x)-plot for the requested results.
            Options for displayed results: scope score ("scope"), vendi score ("vendi"), weighted objectives ("objective", normalized if multiple objectives),
            or individual objectives requested by their respective name.

            Inputs:   
                budget: int
                    experimental budget used in the runs 
                type_results: str
                    Requested type of result.
                    Options:
                        "vendi": Vendi score
                        "objective": average objective values
                        "scope": scope score
                        You can also provide the name of a specific objective for that run and then it will only analyze that one.
                name_results: str
                    Name of the subfolder in which the result csv files are saved.
                scope_method: str
                    method to calculate the scope score ("average","product" (Default),"geometric_mean" )
                objective_mode: dict
                    Dictionary of objective modes for objectives
                    Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
                    Code will assume maximization for all non-listed objectives
                    Default is {"all_obj":"max"}
                objective_weights: dict or None
                    Weights for averaging the objective data
                    Example: {"yield": 0.3, "ee": 0.7}
                    Default is None --> all objectives will be averaged with equal weights.
                bounds: dict
                    dictionary of bounds for the individual metrics (vendi, all objectives)
                    the dict keys are the metric, the values are a tuple of max and min value
                    Default values are for the ArI dataset.
                filename_figure: str or None
                    name for the figure that is created. Default is None --> figure is not saved.
                directory: str
                    current directory. Default: '.'
                show_plot: Boolean
                    Option to display the generated line plot (default is True).
                show_stats: str or None
                    Option to display a statistic to the metric.
                    Options:
                        "stdev": Standard deviation.
                        "min-max": Max and Min values.
                    Default is None.
                specified_batch_size: int or None
                    the code normally infers the batch size from the filename. If a file is name anormal
                    so that this is not possible, the batch size can be specified with this variable.
                    Default is None --> infer the batch size from the filename
            """

            wdir = Path(directory)

            # Get the overview for the requested data
            dfs_scaled,_ = self.get_metric_overview(bounds, budget, name_results, type_results, scope_method, objective_mode,
                            objective_weights, directory, specified_batch_size)

            # Plot and save the figure if requested.
            if show_plot:
                means = dfs_scaled["means"]
                stds = dfs_scaled["stdev"]
                maxs = dfs_scaled["max"]
                mins = dfs_scaled["min"]

                fig, ax = plt.subplots(figsize=(6,6))
                
                for i,col in enumerate(means.columns):
                    x    = means.index.values
                    mean = means[col].values
                    std  = stds[col].values
                    maxval = maxs[col].values
                    minval = mins[col].values

                    # mask nan values that would hinder printing
                    mask = ~np.isnan(mean) & ~np.isnan(std) & ~np.isnan(maxval) & ~np.isnan(minval)
                    x, mean, std, maxval, minval = x[mask], mean[mask], std[mask], maxval[mask], minval[mask]

                    color = self.all_colors[i]
                    ax.plot(x, mean, label=col, color=color)
                    if show_stats == "stdev":
                        ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=color)
                        ax.plot(x, mean + std, linestyle="--", color=color, alpha=0.3)
                        ax.plot(x, mean - std, linestyle="--", color=color, alpha=0.3)
                    elif show_stats == "min-max":
                        ax.fill_between(x, minval, maxval, alpha=0.1, color=color)
                        ax.plot(x, maxval, linestyle="--", color=color, alpha=0.3)
                        ax.plot(x, minval, linestyle="--", color=color, alpha=0.3)
                
                ax.set_xlabel("Scope size")
                if type_results == "comb_obj":
                    type_results = "Combined objective"
                ax.set_ylabel(f"{type_results} score")
                ax.legend(title="Columns")
                plt.tight_layout()
                plt.show()

                # save the figure if requested
                if filename_figure is not None:
                    figure = fig.get_figure()
                    figure.savefig(wdir.joinpath(filename_figure))

            return dfs_scaled  # return the data


    def track_samples(self, filename_umap, filename_data, name_results, scope_method="product", objective_mode = {"all_obj":"max"},
                      objective_weights = None, obj_plot_bounds = None, cbar_scaling = None, cbar_title = None,
                      bounds = {"rate": (2.349,1.035), "vendi": (6.366,1.941)}, display_cut_samples = True, obj_to_display = None, 
                      dpi = 100, figsize = (10,8), size_scaling = 1, filename_labelled = None, show_colorbar = True,
                      rounds_to_display = None, label_round = False, filename_figure = None, hide_axis = False,
                      restrict_samples = None, directory='.'):
        """
        Visually tracks the evaluated and cut samples of a single benchmarking run on a provided UMAP.
        Saves the generated plot. Also provides the results for the run.
        ------------------------------------------------------------------------------------------------
        Inputs:
            filename_umap: str
                name of the file containing the UMAP coordinates
            filename_data: str
                name of the benchmarking run to be analyzed
            name_results: str
                subfolder in which the results are located
            scope_method: str
                method to calculate the scope score ("average","product" (Default),"geometric_mean")
            objective_mode: dict
                Dictionary of objective modes for objectives
                Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
                Code will assume maximization for all non-listed objectives
                Default is {"all_obj":"max"}
            objective_weights: dict or None
                Weights for averaging the objective data
                Example: {"yield": 0.3, "ee": 0.7}
                Default is None --> all objectives will be averaged with equal weights.
            obj_plot_bounds: tuple or None
                option to provide bounds (max,min) for the color bar (default is None)
            cbar_scaling: int or None
                option to scale the values on the colorbar (e. g. if yield scale is [0,1] instead of [0,100])
            cbar_title: str or None
                If None, the objective name will be displayed on the color bar.
                If a string is provided, the color bar label will be set by the variable.
            bounds: dict
                dictionary of bounds for the individual metrics (vendi, all objectives)
                Default values are for the ArI dataset.
            display_cut_samples: Boolean
                show the samples that have been cut by the vendi scoring or not. Default = True.
                If True, the samples in the plot will be colored by the round when they were selected.
                If False, the samples will be displayed by their objective value 
                (default is first listed objective - can be changed in variable obj_to_display)
            obj_to_display: str or None
                color the selected points by objective values if display_cut_samples is False.
                Default is None (take the first listed objective).
            dpi: int
                resolution of the displayed figure
                Default is 100 (also matplotlib default).
            figsize: tuple of 2 int
                figure size.
                Default is (10,8).
            size_scaling: float
                Variable to scale the size of the displayed points (to account for different figsizes).
                Default is 1.
            filename_labelled: str or None
                name of the csv file containing the labelled data for the entire searchspace
                If specified (default is None) and display_cut_samples is False, the pending/cut samples will be
                colored by their theoretical performance.
            show_colorbar: Boolean
                Option to show a colorbar (default is True).
            rounds_to_display: int or None
                Specify how many rounds of the run you want to display (starting from the first one).
                The metrics will also only be calculate for the these rounds.
                E. g.: rounds_to_display=4 --> first 4 rounds will be displayed
                Default is None --> shows all rounds
            label_round: Boolean
                label the suggested samples by the round of selection. Default = False.
            filename_figure: str or None
                name for the figure that is generated
                If None, the figure will not be saved.
            hide_axis: Boolean
                hide the axis ticks if True (Default is False).
            restrict_samples: str
                Option to restrict the UMAP samples to the samples were in the actually used searchspace
                (e. g. when only a subset of a dataset was used for a run)
                The input parameter is the filename of the used searchspace 
                (must be a subset of the one in the UMAP)
                Default is None (no restriction applied).
            directory: str
                current directory. Default: "."
        """

        # Set directory.
        wdir = Path(directory)

        # Read in UMAP and data.
        df_umap = pd.read_csv(wdir.joinpath(filename_umap),index_col=0, float_precision = "round_trip")
        df_data = pd.read_csv(wdir.joinpath(name_results+"/"+filename_data),index_col=0, float_precision = "round_trip")

        # Chec if the UMAP samples will be restricted
        if restrict_samples is not None:
            df_space = pd.read_csv(wdir.joinpath(restrict_samples),index_col=0, float_precision = "round_trip")
            # Only keep the samples at are in the search space
            df_umap = df_umap.loc[df_umap.index.intersection(df_space.index)]
        
        # Prune the number of rounds if requested
        if rounds_to_display is not None:
            df_data = df_data.iloc[:rounds_to_display,:]
        df_data["eval_samples"] = df_data["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
        df_data["cut_samples"] = df_data["cut_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
        # get the objectives
        objectives = ast.literal_eval(df_data.columns[0][11:])
        
        # Get the overall results for this run:
        vendi_score = df_data['Vendi_score'].iloc[-1]
        dict_obj_values = {}
        df_data[f"obj_values {objectives}"] = df_data[f"obj_values {objectives}"].apply(ast.literal_eval)

        for i, obj in enumerate(objectives):
            # get a list with the objective values of each entry
            if len(objectives) > 1:
                dict_obj_values[obj] = [value for round_list in [df_data.loc[round,f"obj_values {objectives}"][i] for round in df_data.index] for value in round_list]
            else:
                dict_obj_values[obj] = [value for round_list in [df_data.loc[round,f"obj_values {objectives}"] for round in df_data.index] for value in round_list]

        # check if one of the objectives was a minimization task
        min_obj = [obj for obj, value in objective_mode.items() if value == "min"]
        # Scale the individual objective values
        dict_obj_scaled = {}
        dict_obj_av = {}
        for obj in dict_obj_values:
            dict_obj_scaled[obj] = [self.normalization(score=value, bounds=bounds[obj]) for value in dict_obj_values[obj]]
            # check if the objective was a minimization task and treat appropriately if so
            if obj in min_obj:
                dict_obj_scaled[obj] = [1 - value for value in dict_obj_scaled[obj]]
            dict_obj_av[obj] = np.mean(dict_obj_scaled[obj])
        
        # calculate the average value for each objective (using weights if submitted)
        if (objective_weights is not None) and (len(objectives) > 1):
                # ensure that the weights sum up to 1
                sum_weights = sum(objective_weights.values())
                objective_weights = {obj: value/sum_weights for obj,value in objective_weights.items()}
        else:
            objective_weights = {obj: 1/len(objectives) for obj in objectives}
        av_obj = sum(objective_weights[obj] * dict_obj_av[obj] for obj in objectives)

        # Scaling the vendi data for the scope score calculation.
        vendi_scaled = self.normalization(score=vendi_score, bounds=bounds["vendi"])
        scope_score  = self.calculate_scope_score(av_obj,vendi_scaled,scope_method)
        print(f"Scope score: {scope_score:.3f}")
        if len(objectives) > 1:
              print(f"Average objective (scaled on [0,1]-scale): {av_obj:.3f}")
        for obj in objectives:
            print(f"Average {obj}: {np.mean(dict_obj_values[obj]):.3f}")
        print(f"Vendi score: {vendi_score:.3f}")
        # Add columns and reassign the index.
        df_umap["status"] = "neutral"
        df_umap["round"] = 0
        obj_plot_name = None
        if obj_to_display is None:
            obj_plot_name = objectives[0]
            obj_plot = dict_obj_values[obj_plot_name]
            if obj_plot_bounds is None:
                obj_plot_bounds = (max(obj_plot),min(obj_plot))
        else:
            obj_plot_name = obj_to_display
            obj_plot = dict_obj_values[obj_plot_name]
            if obj_plot_bounds is None:
                obj_plot_bounds = (max(obj_plot),min(obj_plot))

        df_umap.index = df_umap.index.astype(str)
        # Assign the samples to the df_umap dataframe.
        for round in list(df_data.index):            
            for sample in df_data.loc[round,"eval_samples"]:
                sample = str(sample.encode().decode('unicode_escape'))
                df_umap.loc[sample,"status"] = "suggested"
                df_umap.loc[sample,"round"] = round+1
                df_umap.loc[sample,obj_plot_name] = obj_plot.pop(0)  # this works because samples and objective values are in the same order

            if display_cut_samples:
                for sample in df_data.loc[round,"cut_samples"]:
                    sample = sample.encode().decode('unicode_escape')
                    df_umap.loc[sample,"status"] = "removed"
                    df_umap.loc[sample,"round"] = round+1 

        # Sort df_umap so that important points will be plotted last and won't be covered by neutral points.
        df_umap.sort_values("status",inplace=True,ascending=True)

        # Plot the results
        plt.figure(figsize=figsize,dpi=dpi, constrained_layout=True)

        if display_cut_samples:  # color by round
            colormap = self.cont_cmap2
            norm = mpl.colors.Normalize(vmin=1, vmax=len(df_data.index))  # Normalize the colorscale

            df_selected =df_umap[df_umap["status"] == "suggested"]
            df_pending = df_umap[df_umap['round'] == 0]
            df_cut = df_umap[df_umap["status"] == "removed"]

            plt.scatter(
                df_pending["UMAP1"], df_pending["UMAP2"], s=40*size_scaling, 
                linewidth=0.3*size_scaling,edgecolor="k",
                color=self.all_colors[6], marker="o", alpha = 0.6, zorder=1)

            # Plot the cut points
            plt.scatter(
                df_cut["UMAP1"], df_cut["UMAP2"], s=100*size_scaling, edgecolor="k", marker = "X",
                c=df_cut["round"], cmap=colormap, alpha=0.7, linewidth = 0.3*size_scaling, zorder=2)

            # Plot the selected points
            plt.scatter(df_selected["UMAP1"],df_selected["UMAP2"],c=df_selected["round"],
                                cmap=colormap,norm=norm,s=250*size_scaling,alpha=1,edgecolor='k',
                                linewidth=3*size_scaling, zorder=3)

            if show_colorbar:
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])  # Empty array for ScalarMappable
                cbar = plt.colorbar(sm)
                max_round = int(df_umap["round"].max())
                cbar.set_ticks(range(1, int(max_round)+1, 2))
                cbar.set_ticklabels([str(x) for x in range(1, int(max_round)+1, 2)])
                cbar.set_label('Round')

        else:  # color by obj values
            # Separate selected and non-selected points
            df_umap[obj_plot_name] = df_umap[obj_plot_name].astype(float)
            df_umap.index = df_umap.index.astype(str)
            df_selected =df_umap.loc[df_umap["status"] == "suggested"]
            df_pending =df_umap.loc[df_umap["status"] != "suggested"].copy()

            # Define colormap and normalization
            vmin = obj_plot_bounds[1]
            vmax = obj_plot_bounds[0]
            norm = plt.Normalize(vmin, vmax)
            cmap = self.cont_cmap

            # # Plot non-selected samples
            if filename_labelled is None:
                # grey points if no labels are provided
                plt.scatter(df_pending["UMAP1"], df_pending["UMAP2"], color=self.all_colors[6], s=40*size_scaling, 
                            alpha=0.6, linewidth=0.3*size_scaling, edgecolor="k")
            else:
                # color by their theoretical performance
                df_labelled = pd.read_csv(filename_labelled, index_col = 0, header = 0)
                df_labelled.index = df_labelled.index.astype(str)
                for idx in df_pending.index:
                    idx = str(idx)
                    df_pending.loc[idx,obj_plot_name] = df_labelled.loc[idx,obj_plot_name]
                plt.scatter(df_pending["UMAP1"], df_pending["UMAP2"], cmap=cmap,norm=norm,c=df_pending[obj_plot_name], s=40*size_scaling, 
                        alpha=0.6, linewidth=0.3*size_scaling, edgecolor="k")

            # Plot numeric entries
            scatter_numeric = plt.scatter(df_selected["UMAP1"],df_selected["UMAP2"],c=df_selected[obj_plot_name],
                                          cmap=cmap,norm=norm,s=250*size_scaling,alpha=1,edgecolor='k',linewidth=2*size_scaling)

            # Add colorbar
            if show_colorbar:
                cbar = plt.colorbar(scatter_numeric)
                if cbar_scaling is not None:
                    tick_labels = [f"{int(t * cbar_scaling)}" for t in cbar.get_ticks()]
                    cbar.set_ticklabels(tick_labels)
                if cbar_title is None:
                    cbar_label = obj_plot_name.capitalize()
                    if obj_plot_name.lower() == "yield":
                        cbar_label = "Yield (%)"
                    cbar.set_label(cbar_label)
                else:
                    cbar.set_label(cbar_title)

        if label_round:  # label the round of selection if requested
            texts = []
            for i in df_umap[df_umap["status"]== "suggested"].index:
                texts.append(plt.text(df_umap.loc[i,'UMAP1'], df_umap.loc[i,'UMAP2'], int(df_umap.loc[i,'round']), 
                                      size='medium', color='black', weight='semibold'))
            adjust_text(texts,expand_points=(1.3,1.3),force_static=(10,10),arrowprops={"arrowstyle":"-","color":"black"})
        
        if hide_axis:
            plt.xticks([])
            plt.yticks([])

        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.show()

        if filename_figure is not None:
            figure = plt.get_figure()
            
            figure.savefig(wdir.joinpath(filename_figure))
            
    
    def show_scope(self, filename_data, name_results, by_round = True, rounds_to_display = None, common_core = None,
                   give_data = False, suppress_figure = False, directory = '.', molsPerRow = 6, scale_values = 1, label_suffix = "", round_values = None):
        """
        Depict the substrates that were selected for the scope.
        NOTE: The function is only implemented for singe-objective scenarios.
        ------------------------------------------------------------------------------------------------
        Inputs:
            filename_data: str
                name of the benchmarking run to be analyzed
            name_results: str
                subfolder in which the results are located
            by_round: Boolean
                Select if the selected compounds are shown by round (True --> Default) or all together
            rounds_to_display: int or None
                Specify how many rounds of the run you want to display (starting from the first one).
                The metrics will also only be calculate for the these rounds.
                E. g.: rounds_to_display=5 --> first 4 rounds will be displayed
                Default is None --> shows all rounds
            common_core: str or None
                SMARTS string for the common core of the molecules to align them
                Default: None
            give_data: Boolean
                return the sample dictionary if True (default: False)
            suppress_figure: Boolean
                option to suppress printing of the scope figure (default= False)
            directory: str
                current directory. Default: "."
            molsPerRow: int
                number of molecules per row if by_round is False
            scale_values: int
                Option to scale the result values (e. g. if yields are given on a [0,1]-scale but should be displayed on [0,100]-scale)
                Default is 1 (no scaling).
            label_suffix: str
                label (e. g. unit) to be added to the result values
                Default is an empty string (no label)
            round_values: int or None
                round the objective values to the indicated number of decimal digits if variable is not None (Default). 
                The rounding is applied after scaling.
        """

        # Set directory.
        wdir = Path(directory)

        # Read in UMAP and data.
        df_data = pd.read_csv(wdir.joinpath(name_results+"/"+filename_data), index_col=0,header=0)

        # Prune the number of rounds if requested
        if rounds_to_display is not None:
            df_data = df_data.iloc[:rounds_to_display,:]
        df_data["eval_samples"] = df_data["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])

        # get the objectives
        objectives = ast.literal_eval(df_data.columns[0][11:])
        df_data[f"obj_values {objectives}"] = df_data[f"obj_values {objectives}"].apply(ast.literal_eval)

        # Get a dictionary that maps the samples to the obj_values.
        sample_dict = {}
        for _, row in df_data.iterrows():
            labels = row["eval_samples"]
            values = row[f"obj_values {objectives}"]
            sample_dict.update(dict(zip(labels, values)))
        
        def _generate_representation(smiles_list):
            """"Generate aligned 2D representations of molecules from SMILES strings."""
            # Convert to molecules
            mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

            # Generate 2D coordinates for all mols
            for mol in mol_list:
                AllChem.Compute2DCoords(mol)

            if common_core is not None:
                # Define iodobenzene substructure for alignment
                core = Chem.MolFromSmarts(common_core)
                AllChem.Compute2DCoords(core)

                # Align each molecule to the substructure if it matches
                for mol in mol_list:
                    if mol.HasSubstructMatch(core):
                        AllChem.GenerateDepictionMatching2DStructure(mol, core)

            return mol_list
        
        if not suppress_figure:  # only generate the figure if not suppressed
            if by_round:
                for scope_round in df_data.index:
                    smiles_list = df_data.loc[scope_round,"eval_samples"]
                    mol_list = _generate_representation(smiles_list)
                    # Draw the aligned molecules
                    print(f"Molecules selected in round {scope_round+1}:")
                    if round_values is None:
                        legends = [f"{sample_dict[smiles]*scale_values}"+label_suffix for smiles in smiles_list]
                    else:
                        legends = [f"{round(sample_dict[smiles]*scale_values,round_values)}"+label_suffix for smiles in smiles_list]
                    depiction = Draw.MolsToGridImage(
                        mol_list,
                        molsPerRow=len(mol_list),
                        subImgSize=(200, 200),
                        legends = legends
                        )
                    display(depiction)

            else:
                smiles_list = [smiles for round_list in [df_data.loc[round,"eval_samples"] for round in df_data.index] for smiles in round_list]
                mol_list = _generate_representation(smiles_list)
                if round_values is None:
                    legends = [f"{sample_dict[smiles]*scale_values}"+label_suffix for smiles in smiles_list]
                else:
                    legends = [f"{round(sample_dict[smiles]*scale_values,round_values)}"+label_suffix for smiles in smiles_list]
                depiction = Draw.MolsToGridImage(
                    mol_list,
                    molsPerRow=molsPerRow,
                    subImgSize=(200, 200),
                    legends = legends
                    )
                display(depiction)
        
        if give_data:
            return sample_dict


    @staticmethod
    def feature_analysis(filename,
                         filename_labelled,
                         objectives,
                         objective_mode = {"all_obj":"max"},
                         filename_shap="df_shap.csv",
                         plot_type=["bar"],
                         directory="."):
        
        """
        Wrapper for the ScopeBO().feature_analysis() function to get feature importance using SHAP
        from a Benchmark().collect_data() raw data results file.
        ---------------------------------------------------------------------
        Inputs:
            filename: str
                filename of the reaction space csv file including experimental outcomes
                default is reaction_space.csv
            filename_labelled: str
                filename of the fully labelled reaction space csv file
            objectives: list
                list of the objectives. E. g.: [yield,ee]
            objective_mode: dict
                Dictionary of objective modes for objectives
                Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
                Code will assume maximization for all non-listed objectives
                Default is {"all_obj":"max"}
            filename_shap: str
                The function generates a csv file as input for the ScopeBO().feature_analysis() function.
                Parameter defines the name of that file.
                Default: "df_shap.csv"
            plot_type: list of str
                type of SHAP plot to be generated. Options:
                    "bar" - bar plot of mean absolute SHAP values (Default)
                    "beeswarm" - beeswarm plots of the individual SHAP values
                    both options can be requested by using plot_type = ["bar","beeswarm"]
                    providing an empty list supresses plotting
            directory: str
                Define working directory. Default is current directory (".").

        ---------------------------------------------------------------------
        Returns the shap.explainer object, mean absolute SHAP values, and the requested plot of SHAP values.
        """

        wdir = Path(directory)

        # Load the results file
        df_results = pd.read_csv(wdir.joinpath(filename),index_col=0,header=0)
        # Process the data
        df_results["eval_samples"] = df_results["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])

        # Get the selected samples
        idx_samples = []
        for idx in df_results.index:
            for sample in df_results.loc[idx,"eval_samples"]:
               idx_samples.append(sample.encode().decode('unicode_escape'))

        # Load the labelled dataframe and remove the objective data for all samples that have not been selected
        df_shap = pd.read_csv(wdir.joinpath(filename_labelled),index_col=0,header=0)
        idx_pending = df_shap.index.difference(idx_samples)
        for objective in objectives:
            df_shap.loc[idx_pending, objective] = "PENDING"

        # Save the dataframe
        df_shap.to_csv(wdir.joinpath(filename_shap),index=True,header=True)

        # Call the ScopeBO() function for feature analysis
        shap_values, mean_abs_shap_values = ScopeBO().feature_analysis(objectives=objectives,
                                   objective_mode=objective_mode,
                                   filename=filename_shap,
                                   plot_type=plot_type,
                                   directory=directory)
        
        return shap_values,mean_abs_shap_values
    

    def objective_distribution(self,name_results, objective_bounds = (0,100), nr_bins = 10, norm_axis = None, directory = ".", print_figure = True):
        """
        Compute and visualize the distribution of objective values across the different random seeds.
        NOTE: Currently only supports single-objective optimization.

        Parameters
        ----------
        name_results : str
            Path to the result folder containing the subfolder ``raw_data`` with 
            CSV files of raw optimization data.
        objective_bounds : tuple of (float, float)
            Lower and upper bounds of the objective value range for binning.
            Default is (0, 100).
        nr_bins : int
            Number of equally spaced bins to divide the objective range into.
            Default is 10.
        norm_axis : int or None
            Max value for the histogram count axis.
            If None (Default), the axis max value is determined automatically.
        directory : str
            Working directory. Default is current directory.
        print_figure : bool
            If True, generate a bar plot showing the average counts per bin and
            error bars corresponding to the standard deviation across runs.
            Default is True.

        Returns
        -------
        df_counts : pandas.DataFrame
            A dataframe where each row corresponds to one run and each column 
            contains the counts of objective values falling into each bin.
        """

        wdir = Path(directory)
        raw_path = wdir / name_results / "raw_data"

        # find the objectives
        objectives = Benchmark.find_objectives(name_results,directory)

        # define the bins
        bins = np.linspace(objective_bounds[0],objective_bounds[1],nr_bins+1)

        # list to store the count results
        counts = []

        # go through all the raw files in the original folder
        for file in os.listdir(raw_path):

            # read in the raw data file
            df_raw = pd.read_csv(raw_path / file, index_col = 0, header = 0)

            # get the objective values of each entry
            df_raw[f"obj_values {objectives}"] = df_raw[f"obj_values {objectives}"].apply(ast.literal_eval)
            obj_values = [value for round_list in [df_raw.loc[round,f"obj_values {objectives}"] for round in df_raw.index] for value in round_list]

            # get the counts for the bins
            counts.append(np.histogram(obj_values,bins=bins)[0])

        # convert to df
        df_counts = pd.DataFrame(counts).fillna(0)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        df_counts.columns = bin_centers

        # print the average distribution and its standard deviation if requested
        if print_figure:
            mean_counts = df_counts.mean()
            std_counts = df_counts.std(ddof=1)  # using Bessel's correction

            plt.figure(figsize=(8,6))
            plt.bar(bin_centers, mean_counts, width=(bins[1]-bins[0]), yerr=std_counts, 
                    capsize=5, alpha=1, color=self.all_colors[1], edgecolor='k',zorder =2)
            plt.xlabel(f'{objectives[0].capitalize()} value')
            plt.ylabel('Average Count')
            if norm_axis is not None:
                plt.ylim(0, norm_axis)
            plt.title('Average Objective Distribution with Standard Deviation')
            plt.xticks(bins)
            plt.grid(axis='y',zorder=1)
            plt.show()
        
        return df_counts


    @staticmethod
    def normalization(score,bounds):
        """
        Helper function:
        Function to normalize Vendi scores and average objectives.
        Input:
            score: float or int
                result to be processed
            bounds: tuple
                upper and lower bounds (in that order) of the scores for the normalization
        """
        return (score-bounds[1])/(bounds[0]-bounds[1])



    @staticmethod
    def standardization(score,distr_metrics):
        """
        Helper function:
        Standardizes a provided score.
        distr_metric: tuple
            (mean, stdev) --> both type float
        """
        return (score-distr_metrics[0])/distr_metrics[1]
    

    @staticmethod
    def calculate_scope_score(obj_score, vendi_score, method = "product"):
        """
        Helper function:
        Calculate the scope score using different calculation methods.
        obj_score, vendi_score: float of the respective scaled score.
        method: string (Options "average", "product" (default), "geometric_mean")
        """
        scope_score = None
        if "av" in method.lower():
            scope_score = (obj_score + vendi_score) / 2
        elif "prod" in method.lower():
            scope_score = obj_score * vendi_score
        elif "geo" in method.lower():
            prod = obj_score * vendi_score
            # Taking the root is only possible for positive numbers.
            # Define the prod as 0 in case it is negative 
            # so that the scope score will be 0 as well
            if isinstance(prod, pd.DataFrame):
                prod = prod.where( (prod > 0) | prod.isna(), 0)
            elif prod < 0:
                prod = 0
            scope_score = np.sqrt(prod)

        return scope_score
              
    
    @staticmethod
    def find_objectives(folder_name,directory):

        """
        Helper function to find the names of the objectives in the raw results.
        -------------------
        Input:
            folder_name: path of the folder to be analzyed
        directory: 
            working directory
        -------------------
        Returns:
            list of the objectives that were used in this run
        """

        wdir = Path(directory)
        # Get a list of the raw files of the run
        raw_path = wdir.joinpath(folder_name+"/raw_data/")
        raw_files = os.listdir(raw_path)
        # Load one of the files
        df_results = pd.read_csv(wdir.joinpath(raw_path,raw_files[0]),index_col=0,header=0)
        
        # Get the list of objectives from the column name and return it
        return ast.literal_eval(df_results.columns[0][11:])
    

    @staticmethod
    def get_metric_overview(bounds, budget, name_results, type_results, scope_method = "product", 
                            objective_mode = {"all_obj":"max"}, objective_weights = None, directory = ".",
                            specified_batch_size = None):
        """
        Compute a metric summary across seeds and hyperparameter settings for use in
        different plotting functions.

        This function aggregates raw result files produced by optimization runs,
        groups them by hyperparameter combinations, extracts objective values and
        Vendi scores, converts round-level metrics into cumulative metrics, and
        computes mean, standard deviation, min, and max across seeds. It also
        constructs per-seed raw DataFrames and aligns results to the evaluation
        budget.

        Parameters
        ----------
        bounds: dict
            dictionary of bounds for the individual metrics (vendi, all objectives)
            the dict keys are the metric, the values are a tuple of max and min values
        budget : int
            experimental budget
        name_results: str
            name of the folder in which the results will be saved
                type_results: str
                    Requested type of result.
                    Options:
                        "vendi": Vendi score
                        "objective": average objective values
                        "scope": scope score
                        You can also provide the name of a specific objective for that run and then it will only analyze that one.
            If multiple objectives are present and "objective" is requested,
            a combined objective score is computed.
        scope_method: str
            method to calculate the scope score ("average","product" (Default),"geometric_mean" )
        objective_mode: dict
            Dictionary of objective modes for objectives
            Provide dict with value "min" in case of a minimization task (e. g. {"cost":"min"})
            Code will assume maximization for all non-listed objectives
            Default is {"all_obj":"max"} --> all objectives are maximized   
        objective_weights: list or None
            list of float weights for the scalarization of the objectives 
            only relevant for multi-objective greedy runs, not other acquisition functions
            Default: None (objectives will be averaged)
        directory: str
            working directory (default is current directory)
        specified_batch_size : int or None, optional
            If provided, each round is assumed to use this fixed batch size.
            If None, batch sizes are inferred from the hyperparameter setting
            encoded in the filename.

        Returns
        -------
        dfs_scaled : dict of pandas.DataFrame
            Summary dataframes for the requested metric, aggregated across seeds:
            - "means" (mean values)
            - "stdev" (standard deviation with Bessel correction)
            - "max" (max values)
            - "min" (min values) 
            Each DataFrame has:
                - rows indexed by experiment number (1..budget)
                - columns indexed by hyperparameter combinations

        dict_dfs_raw_data : dict of dict of pandas.DataFrame
            Nested dictionary:
            - Outer key: hyperparameter combination
            - Inner key: seed index
            - Inner value: raw per-seed DataFrame containing cumulative objective
            values, Vendi scores, combined objectives, and scope scores (where applicable).

        NOTE: Files must follow the naming convention "<prefix>_<batch>_<setting>_s<seed>.csv"  
        """

        # get all the raw files and sort them by hyperparameter combination
        wdir = Path(directory)
        raw_path = wdir.joinpath(name_results+"/raw_data/")
        setting_dict = {}
        for file in os.listdir(raw_path):
            combi = "_".join(file.split("_")[1:3])
            if "low_variance" in file:  # special case for low_variance acq fct that has an underscore in its name
                combi = "_".join(file.split("_")[2:4])
            if combi not in setting_dict.keys():
                setting_dict[combi] = [file]
            else:
                setting_dict[combi].append(file)

        # get the objective names
        objectives = Benchmark.find_objectives(name_results,directory)

        # go through all the different hyperparameter settings
        batch_sizes_list = []
        dict_unscaled_mean = {}
        dict_unscaled_stdev = {}
        dict_unscaled_max = {}
        dict_unscaled_min = {}
        dict_dfs_raw_data = {}
        for combi in setting_dict.keys():

            # list to store the results of all seeds for one setting
            seeded_list = []
            seeds = len(setting_dict[combi])

            # go through the different files
            # (sort them first so that they are saved in the correct order in the returned raw data)

            # Regex to extract seed number
            seed_pattern = re.compile(r's(\d+)\.csv$')
            # Sort the files by their random seed
            sorted_files = sorted(setting_dict[combi], key=lambda x: int(seed_pattern.search(x).group(1)))

            # loop through the individual seeds
            for seed_file in sorted_files:

                # get lists of the values in each round for each objective and the vendi score (for one seed)
                dict_raw_data = {}
                batch_sizes = None

                # read in the raw data
                df_raw = pd.read_csv(f"{raw_path}/{seed_file}",index_col=0,header=0)
                df_raw[f"obj_values {objectives}"] = df_raw[f"obj_values {objectives}"].apply(lambda x: ast.literal_eval(x))

                #get the vendi data
                dict_raw_data["vendi"] = df_raw["Vendi_score"].to_list()

                # get the objective data
                for i,obj in enumerate(objectives):
                    if len(objectives) > 1:  # structure: [[<values obj1>],[<values obj2>]]
                        dict_raw_data[obj] = df_raw[f"obj_values {objectives}"].apply(lambda x: np.mean(x[i])).to_list()
                    else:  # structure: [<values obj1>]
                        dict_raw_data[obj] = df_raw[f"obj_values {objectives}"].apply(lambda x: np.mean(x)).to_list()
                    # figure out the batch sizes if not specified
                    if specified_batch_size is None:
                        batch = combi.split("_")[0][1:]
                        if batch.isdigit():  # case using a fixed batch size
                            batch = int(batch)
                            batch_sizes = [int(batch)]*len(dict_raw_data[obj])
                            difference = budget - sum(batch_sizes)
                            batch_sizes[-1] += difference  # reduce the last batch if it was smaller due to budget constraints
                        else:  # case using different batch sizes in each round
                            batch_sizes = [int(el) for el in batch.split("-")]  # list with the batch sizes for each round
                    else:
                        batch_sizes = [specified_batch_size] * len(dict_raw_data[obj])
                    # the obj data is the average value obtained in each round, 
                    # but we need the culmulative results until this round
                    total_obj = [i*j for i,j in zip(batch_sizes,dict_raw_data[obj])]  # batch size * average obj for each round
                    processed_obj = []
                    for round in range(len(dict_raw_data[obj])):
                        processed_result = sum(total_obj[:(round+1)]) / sum(batch_sizes[:(round+1)])
                        processed_obj.append(processed_result)
                    # reassign the obj data (now with the cumulative averages)
                    dict_raw_data[obj] = processed_obj

                # save the results for this seed
                seeded_list.append(dict_raw_data)

            batch_sizes_list.append(batch_sizes)

            # process the requested data
            processed_mean = []
            processed_stdev = []
            processed_max = []
            processed_min = []

            # generated a copy of the seeded_list that will not be further processed and can be returned
            # (can loop through a range object for the keys because the result files have been analyzed in sorted order)
            dfs_raw_data = {
                i: pd.DataFrame(seeded_list[i],
                                index = range(len(seeded_list[i]["vendi"])),
                                columns=seeded_list[i].keys()) 
                for i in range(len(seeded_list))
                }
            dict_dfs_raw_data[combi] = dfs_raw_data

            # case when a specific objective (or the only one) or the vendi score is requested
            if type_results.lower() not in ["scope","objective"] or (type_results.lower() == "objective" and len(objectives) == 1):
                if type_results.lower() == "objective":
                    type_results = objectives[0]  # reassign with the name of the only objective
                stacked = np.stack([d[type_results.lower()] for d in seeded_list], axis=0)
                processed_mean = np.mean(stacked, axis=0).tolist()
                if seeds > 1:
                    processed_stdev  = np.std(stacked, axis=0, ddof=1).tolist()  # using Bessel's correction
                else:
                    processed_stdev  = np.std(stacked, axis=0).tolist()  # no correction if single seed
                processed_max = np.max(stacked, axis=0).tolist()
                processed_min = np.min(stacked, axis=0).tolist()


            # combined objectives or scope score is requested
            else:
                if type_results.lower() == "objective":
                    type_results = "comb_obj"
                # check for minimization objectives
                min_obj = [obj for obj, value in objective_mode.items() if value == "min"]

                # preprocess the objective weights
                if (objective_weights is not None):
                    # ensure that the weights sum up to 1
                    sum_weights = sum(objective_weights.values())
                    objective_weights = {obj: value/sum_weights for obj,value in objective_weights.items()}
                else:
                    objective_weights = {obj: 1/len(objectives) for obj in objectives}
                # process the data for the individual seeds
                for seed in range(len(seeded_list)):
                    # normalize the data
                    for metric in objectives + ["vendi"]:
                        seeded_list[seed][metric] = [Benchmark.normalization(score=x,bounds=bounds[metric]) for x in seeded_list[seed][metric]]

                    # invert minimization tasks
                    if min_obj:
                        for obj in min_obj:
                                seeded_list[seed][obj] = [1 - x for x in seeded_list[seed][obj]]

                    # combine the objectives
                    seeded_list[seed]["comb_obj"] = sum(objective_weights[obj] * np.array(seeded_list[seed][obj])
                                                        for obj in objectives).tolist()
                    # calculate the scope score
                    if type_results.lower() == "scope":
                        seeded_list[seed]["scope"] = [Benchmark.calculate_scope_score(comb_obj,vendi,scope_method) for comb_obj,vendi in zip(seeded_list[seed]["comb_obj"],seeded_list[seed]["vendi"])]
                        # also record the scope score in dfs_raw_data
                        dfs_raw_data[seed]["scope"] = seeded_list[seed]["scope"]

                stacked = np.stack([d[type_results.lower()] for d in seeded_list], axis=0)
                processed_mean = np.mean(stacked, axis=0).tolist()
                if seeds > 1:
                    processed_stdev  = np.std(stacked, axis=0, ddof=1).tolist()  # using Bessel's correction
                else:
                    processed_stdev  = np.std(stacked, axis=0).tolist()  # no correction if single seed
                processed_max = np.max(stacked, axis=0).tolist()
                processed_min = np.min(stacked, axis=0).tolist()

            dict_unscaled_mean[combi] = processed_mean
            dict_unscaled_stdev[combi] = processed_stdev
            dict_unscaled_max[combi] = processed_max
            dict_unscaled_min[combi] = processed_min

        # ensure all results lists have the same length by appending np.NaN
        max_len = max(len(v) for v in dict_unscaled_mean.values())
        dict_unscaled_mean = {k: v + [np.nan] * (max_len - len(v)) for k, v in dict_unscaled_mean.items()}
        dict_unscaled_stdev = {k: v + [np.nan] * (max_len - len(v)) for k, v in dict_unscaled_stdev.items()}
        dict_unscaled_max = {k: v + [np.nan] * (max_len - len(v)) for k, v in dict_unscaled_max.items()}
        dict_unscaled_min = {k: v + [np.nan] * (max_len - len(v)) for k, v in dict_unscaled_min.items()}

        # convert to dfs
        dfs_unscaled = {}
        dfs_unscaled["means"] = pd.DataFrame.from_dict(dict_unscaled_mean)
        dfs_unscaled["stdev"] = pd.DataFrame.from_dict(dict_unscaled_stdev)
        dfs_unscaled["max"] = pd.DataFrame.from_dict(dict_unscaled_max)
        dfs_unscaled["min"] = pd.DataFrame.from_dict(dict_unscaled_min)

        #  The data is currently shown per batch (which can be different for the different runs). Still needs to be "scaled" to the number of experiments. 
        dfs_scaled = {}
        for metric,df in dfs_unscaled.items():
            dfs_scaled[metric] = pd.DataFrame(np.nan,[x+1 for x in range(budget)],df.columns)  # create empty dataframe with the right shape
            for column_nr in range(len(df.columns)):
                batch_sizes = batch_sizes_list[column_nr]  # batch sizes for each round
                nr_experiments = [sum(batch_sizes[:round+1]) for round in range(len(batch_sizes))]  # total number of experiments including for each round
                for entry in range(len(df.index)):
                    if pd.notna(df.iloc[entry,column_nr]):  # check that the value is actually numeric and not nan
                        dfs_scaled[metric].iloc[nr_experiments[entry]-1, column_nr] = df.iloc[entry,column_nr]  # -1 because iloc is 0-indexed

        # clean up
        for key in dfs_scaled.keys():
            dfs_scaled[key].dropna(how="all",inplace=True)
            dfs_scaled[key] = dfs_scaled[key][sorted(dfs_scaled[key].columns)]

        return dfs_scaled, dict_dfs_raw_data
import ast
import os
from pathlib import Path
from statistics import stdev
import sys

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import seaborn as sns
from adjustText import adjust_text

from scripts.predictor  import ScopeBO


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')


    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Benchmark:
    """Class for generating benchmarking data and analyzing this data."""

    def __init__(self):
        pass

    
    def collect_data(self, filename_labelled, objectives, objective_mode,
            seeds, name_results, init_sampling_method="random", Vendi_pruning_fractions = [None], batches=[None],budget=25, objective_weights = None, sample_threshold = None, 
            enforce_dissimilarity=False, pruning_metric = "vendi_sample", acquisition_function_mode = 'greedy', 
            dft_filename = None, filename_prediction = "df_benchmark.csv", directory='.'):
        """
        Runs the function for selected parameter ranges and records the results per round for each set of parameters.

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
            Summarized data:
                These files contain either objective value data (averaged for each batch round) ("obj") or
                Vendi score data ("vendi"). The rows correspond to the different batch sizes in batches (noted as indices). The columns
                correspond to the different Vendi_pruning_fractions.
                The entries are lists containing the results per round. There are files contain average values ("average") (across the different seeds) or 
                the standard deviation ("stdev") (across the different seeds).
                In the case of multiple objectives, a separate file for each objective value as well as a file containing the combined objective values are generated.

                Example name for a run with the objectives "yield" and "selectivity": "benchmark_obj[yield__selectivity]_av.csv" --> average objective value data
        
        -------------------------------------------------------------------------------------

        filename_labelled: string
            name of the csv file with the labelled data
        filename_prediction: string
            name of the csv file in which the benchmark dataframe is saved
            Default: "df_benchmark.csv"
        objective: list
            list indicating the column name for the objective (string)
        objective_mode: list
            list containing the mode of the objective (string): "max" or "min"      
        budget: int
            experimental budget
        init_sampling_method: str
            sampling method. Options:
                "random": random selection
                "cvt": CVT sampling
                "lhs": LHS sampling
        batches: list
            list of batch sizes (as int numbers)
        seeds: list
            list of seed values
        name_results: str
            name of the folder in which the results will be saved
        Vendi_pruning_fractions: list
            list of threshold values for the Vendi cutoff (floats)  
        dft_filename: None or str
            name of the file containing the dft-featurized reaction space for the vendi score calculation
            only has to be given if the substrate encoding is not dft-derived (e. g. Mordred or Rdkit featurization)
            Default: None
        """
        
        wdir = Path(directory)
        # Create the results folder and the folder for raw results.
        if not os.path.exists(wdir.joinpath(name_results)):
            # Create the folder
            os.makedirs(wdir.joinpath(name_results))
        if not os.path.exists(wdir.joinpath(name_results+"/raw_data")):
            # Create the folder
            os.makedirs(wdir.joinpath(name_results+"/raw_data"))

        # Read labelled data.
        df_labelled = pd.read_csv(wdir.joinpath(filename_labelled),index_col=0,header=0, float_precision = "round_trip")

        # Generate a copy of the DataFrame with labelled data and remove the objective data.
        df_unlabelled = df_labelled.copy(deep=True)
        df_unlabelled.drop(columns=objectives,inplace=True)

        # Instantiate empty df for the analyzed results.
        Vendi_names = []
        for Vpf in Vendi_pruning_fractions:
            if type(Vpf) is list:
                rounded_Vpf = [round(el,1) if type(el) is float else el for el in Vpf]
                Vendi_names.append("-".join(map(str,rounded_Vpf)))
            elif Vpf is None:
                Vendi_names.append("11-39-39-11-11-11-11")
            else:
                Vendi_names.append(str(Vpf))
        batch_names = []
        for batch in batches:
            if type(batch) is list:
                batch_names.append("-".join(map(str,batch)))
            elif batch is None:
                batch_names.append("4-1-1-4-5-5-5")
            else:
                batch_names.append(str(batch))
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
        total_runs = len(batches) * len(Vendi_pruning_fractions) * seeds  # same
        for batch in batches: 
            for Vpf in Vendi_pruning_fractions:
                
                seeded_list_obj = []
                seeded_list_vendi = []

                for seed in range(seeds):     
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
                        if batch is None:
                            rounds = 7  # value using optimized conditions
                        elif budget % batch != 0:
                            rounds = int(budget/batch)+1 # extra round with reduced batch size for last run (will be reduced below)
                        
                        else:
                            rounds = int(budget/batch)
                    
                    # Run ScopeBO for these settings.
                    for current_round in range(rounds):
                            
                        # check if the batch sie is dynamic (meaning different batch sizes for each rounds)
                        current_batch = None
                        if type(batch) is list:
                            current_batch = batch[current_round]
                        else:
                            current_batch = batch
                            # Check if this will be a run with reduced batch size (due to the set budget).
                            if batch is not None:
                                if current_round+1 == rounds and budget % batch != 0:
                                    current_batch = budget % batch

                        # Check if the Vendi_pruning_fraction is dynamic (meaning different fractions for each round)
                        this_Vendi_pruning_fraction = Vpf
                        if type(Vpf) is list:
                            this_Vendi_pruning_fraction = Vpf[current_round]

                        # assign labels for the printout
                        batch_label = batch
                        Vpf_label = this_Vendi_pruning_fraction
                        current_batch_label = current_batch
                        if batch is None:
                            batch_label = "default"
                            batch_sizes_default = [4,1,1,4,5,5,5]
                            current_batch_label = batch_sizes_default[current_round]
                        if this_Vendi_pruning_fraction is None:
                            Vpf_label = "default"

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
                        if vendiScopeBO:  # this is the scenario when a non-dft featurizatio is used in the campaign
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
                    csv_filename_results = wdir.joinpath(name_results+f"/raw_data/{budget}{acquisition_function_mode}_b{batch_names[batches.index(batch)]}_V{Vendi_names[Vendi_pruning_fractions.index(Vpf)]}_s{seed}.csv")
                    if len(df_results.iloc[0,0]) == 1:  # flatten unnecessary list of lists for mono-objective runs
                        for idx in df_results.index:
                            df_results.loc[idx,f"obj_values {objectives}"] = str(df_results.loc[idx,f"obj_values {objectives}"][0])
                    df_results.to_csv(csv_filename_results,index=True,header=True)

                    print (f"Finished campaign {run_counter} of {total_runs}.")
                    run_counter+= 1
                
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



        # Save the dataframes with the processed results as csv files.        
        df_obj_av.to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{'__'.join(objectives)}]_av.csv"),index=True,header=True)
        df_obj_stdev.to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{'__'.join(objectives)}]_stdev.csv"),index=True,header=True)
        df_vendi_av.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index=True,header=True)
        df_vendi_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_stdev.csv"),index=True,header=True)
        if len(objectives) > 1:
            for objective in objectives:
                dfs_indiv_obj_av[objective].to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{objective}]_av.csv"),index=True,header=True)
                dfs_indiv_obj_stdev[objective].to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{objective}]_stdev.csv"),index=True,header=True)


        print(f"Data collection finished! Results are saved in the subfolder {name_results}.")


    def continue_data_collection(self, filename_labelled, objectives, objective_mode, copy_rounds, 
                                 remaining_budget, batches, 
                                 Vendi_pruning_fractions, seeds, name_results, name_prior_results, 
                                 init_sampling_method, sample_threshold = None, enforce_dissimilarity=False, 
                                 pruning_metric = "vendi", acquisition_function_mode = 'balanced', dft_filename = None, 
                                 filename_prediction = "df_benchmark.csv", directory='.'):
        """
        Takes the first n rounds from a different run and continues them with the indicated settings. Otherwise, the same as collect_data().
        See the docstring of collect_data() for full details on the generated reports and most variables.
        NOTE: The function only works if the folder name_prior_results only contains one run (with different seeds)!
        NOTE: The function also only works for mono-objective runs at the moment.
        -------------------------------------------------------------------------------------
        Additional parameters compared to collect_data():

        copy_rounds: int
            number of rounds to take from the previous run (meaning all rounds until the indicated one)
        remaining_budget: int
            number of experiments to be added in this run
        name_results: string
            name for saving the generated results
        name_prior_results: str
            name of the folder from which results are read in
        """
        
        wdir = Path(directory)
        # Create the results folder and the folder for raw results.
        if not os.path.exists(wdir.joinpath(name_results)):
            # Create the folder
            os.makedirs(wdir.joinpath(name_results))
        if not os.path.exists(wdir.joinpath(name_results+"/raw_data")):
            # Create the folder
            os.makedirs(wdir.joinpath(name_results+"/raw_data"))

        # Read labelled data.
        df_labelled = pd.read_csv(wdir.joinpath(filename_labelled),index_col=0,header=0, float_precision = "round_trip")

        # Generate a copy of the DataFrame with labelled data and remove the objective data. NOTE: only implemented for single objective benchmarks.
        df_unlabelled = df_labelled.copy(deep=True)
        df_unlabelled.drop(columns=[objectives[0]],inplace=True)

        # Instantiate empty df for the analyzed results.
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
        df_obj_av = pd.DataFrame(None,batch_names,Vendi_names)
        df_obj_stdev = pd.DataFrame(None,batch_names,Vendi_names)
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
        total_runs = len(batches) * len(Vendi_pruning_fractions) * seeds  # same
        for batch in batches: 
            for Vendi_pruning_fraction in Vendi_pruning_fractions:
                
                seeded_list_obj = []
                seeded_list_vendi = []

                for seed in range(seeds): 

                    current_df = df_unlabelled.copy(deep=True)

                    # read in the prior results with the correct seed
                    prior_run_name = None
                    for filename in os.listdir(wdir.joinpath(name_prior_results+"/raw_data/")):
                        if f"s{seed}" in filename:  # check for the file with the correct seed by using naming convention
                            prior_run_name = filename
                    df_prior_data = pd.read_csv(wdir.joinpath(name_prior_results+"/raw_data/"+prior_run_name),index_col=0,header=0, float_precision = "round_trip")
                                                            
                    # process this df
                    df_prior_data["eval_samples"] = df_prior_data["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
                    df_prior_data["cut_samples"] = df_prior_data["cut_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])

                    # collect the selected samples for all requested rounds
                    prior_selected = [smiles.encode().decode('unicode_escape') for round_list in [df_prior_data.loc[round,"eval_samples"] for round in df_prior_data.index[:copy_rounds]] for smiles in round_list]
                    prior_budget = len(prior_selected)

                    # same for the pruned samples
                    prior_cut = [smiles.encode().decode('unicode_escape') for round_list in [df_prior_data.loc[round,"cut_samples"] for round in df_prior_data.index[:copy_rounds]] for smiles in round_list]

                    # typically there is no pruning in the first round of experiments (initiation), resulting in the list containing an empty element - delete that element
                    if "" in prior_cut:
                        prior_cut.remove("")

                    # get the previously measured objective values
                    prior_obj = list(df_labelled.loc[df_labelled.index.isin(prior_selected),objectives[0]])

                    # assign the priorities in the dataframe
                    current_df["priority"] = 0
                    current_df.loc[prior_selected,"priority"] = -2
                    current_df.loc[prior_cut,"priority"] = -1

                    # Set up lists to hold raw results and average results for this run.
                    raw_results = []
                    run_results = []

                    # Save the prior results in a list and append it to the overall results list.
                    current_raw_results = []
                    current_raw_results.append(prior_obj)
                    current_raw_results.append(df_prior_data.loc[copy_rounds-1,"Vendi_score"])
                    current_raw_results.append(prior_selected)
                    current_raw_results.append(prior_cut)
                    raw_results.append(current_raw_results)
                    prior_obj_av = sum(prior_obj)/len(prior_obj)


                    # Save the processed results for this round.
                    run_results.append([prior_obj_av,df_prior_data.loc[copy_rounds-1,"Vendi_score"]])

                    # assign the objective values in the dataframe
                    current_df[objectives[0]] = "PENDING"
                    for idx in prior_selected:
                        current_df.loc[idx,objectives[0]] = df_labelled.loc[idx,objectives[0]]

                    current_df.to_csv(csv_filename_pred, index=True, header=True)
                    if df_dft is not None:
                        current_df_dft = df_dft.copy(deep=True)  # reset the dft-feautrized df for the vendi calculation
                    
                    # Determine the number of rounds of experiments for the given batch size.
                    rounds = 0
                    if type(batch) is list:
                        rounds = len(batch)
                    else:
                        if remaining_budget % batch != 0:
                            rounds = int(remaining_budget/batch)+1 # extra round with reduced batch size for last run (will be reduced below)
                        
                        else:
                            rounds = int(remaining_budget/batch)

                    # Run ScopeBO for these settings.
                    for current_round in range(rounds):
                            
                        # check if the batch sie is dynamic (meaning different batch sizes for each rounds)
                        current_batch = None
                        if type(batch) is list:
                            current_batch = batch[current_round]
                        else:
                            current_batch = batch
                            # Check if this will be a run with reduced batch size (due to the set budget).
                            if current_round+1 == rounds and remaining_budget % batch != 0:
                                current_batch = remaining_budget % batch

                        # Check if the Vendi_pruning_fraction is dynamic (meaning different fractions for each round)
                        this_Vendi_pruning_fraction = None
                        if type(Vendi_pruning_fraction) is list:
                            this_Vendi_pruning_fraction = Vendi_pruning_fraction[current_round]
                        else:
                            this_Vendi_pruning_fraction = Vendi_pruning_fraction

                        print(f"Now running Batch size: {batch}, Vendi_pruning_fraction: {this_Vendi_pruning_fraction}, Seed: {seed}, Round: {current_round}, current batch: {current_batch}")
                        with HiddenPrints():
                            current_df = myScopeBO.run(
                                objectives = objectives,
                                objective_mode= objective_mode,
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
                        
                        # reset the list to hold the raw results for this round
                        current_raw_results = []

                        # Save indices of samples.
                        current_idx_samples = list(current_df[current_df["priority"]  == 1].index)

                        # Update dataframe with results and save the objective values.
                        current_obj = []
                        for idx in current_idx_samples:
                            current_df.loc[idx,objectives[0]] = df_labelled.loc[idx,objectives[0]]  # NOTE: there is only one objective in the benchmarking dataset
                            current_obj.append(df_labelled.loc[idx,objectives[0]])

                        # Save the dataframe for the next round of ScopeBO.
                        current_df.to_csv(csv_filename_pred, index=True, header=True)

                        # Calculate the Vendi score for all points that were observed so far.
                        if vendiScopeBO:  # this is the scenario when a non-dft featurizatio is used in the campaign
                            for idx in current_idx_samples:
                                current_df_dft.loc[idx,objectives[0]] = df_labelled.loc[idx,objectives[0]]
                            current_df_dft.to_csv(wdir.joinpath(filename_vendi),index=True,header=True)
                            current_vendi_score = vendiScopeBO.get_vendi_score(objectives = objectives, 
                                                                            directory = directory, filename = filename_vendi)
                            print("Vendi score calculated via additional file with dft.")                              
                        else:  # this is the standard case for a campaign using dft featurization
                            current_vendi_score = myScopeBO.get_vendi_score(objectives = objectives, directory = directory, filename = filename_prediction)

                        # Get the newly pruned samples by looking up all pruned samples and removing the ones that were already pruned.
                        current_idx_cut = list(current_df[current_df["priority"]  == -1].index)
                        for i in range(current_round+1):  # loop through the previously saved batches. (plus one because results from prior run have been saved before round 0 of the run)
                            for j in raw_results[i][3]:  # cut samples are saved as the 4th entry in each current_results list
                                current_idx_cut.remove(j)                      

                        # Save results for this round in a list and append it to the overall results list.
                        current_raw_results.append(current_obj)
                        current_raw_results.append(current_vendi_score)
                        current_raw_results.append(current_idx_samples)
                        current_raw_results.append(current_idx_cut)
                        raw_results.append(current_raw_results)

                        # Average the objective value for all samples in this round.
                        current_obj_av = sum(current_obj)/len(current_obj)

                        # Save the processed results for this round.
                        run_results.append([current_obj_av,current_vendi_score])

                    # Save the processed results for this run.
                    seeded_list_obj.append([run_results[i][0] for i in range(len(run_results))])
                    seeded_list_vendi.append([run_results[i][1] for i in range(len(run_results))])
                    
                    # Save raw results as a csv.
                    df_results = pd.DataFrame(raw_results,columns=[f"obj_values {objectives}","Vendi_score","eval_samples","cut_samples"])
                    csv_filename_results = wdir.joinpath(name_results+f"/raw_data/{prior_budget}+{remaining_budget}{acquisition_function_mode}_b{batch_names[batches.index(batch)]}_V{Vendi_names[Vendi_pruning_fractions.index(Vpf)]}_s{seed}.csv")
                    if len(df_results.iloc[0,0]) == 1:  # flatten unnecessary list of lists for mono-objective runs
                        for idx in df_results.index:
                            df_results.loc[idx,f"obj_values {objectives}"] = str(df_results.loc[idx,f"obj_values {objectives}"][0])
                    df_results.to_csv(csv_filename_results,index=True,header=True)

                    print (f"Finished campaign {run_counter} of {total_runs}.")
                    run_counter+= 1
                
                # Calculate the averages and standard deviations across the different seeds and save the results.
                obj_value = [[sum(matrix[i][j] for matrix in seeded_list_obj) / len(seeded_list_obj) for j in range(len(seeded_list_obj[0][0]))] for i in range(len(seeded_list_obj[0]))]
                df_obj_av.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([value for this_round in obj_value for value in this_round])
               
                # Standard deviation can only be calculated if there are at least 2 values. Set to zero if there is only one.
                if seeds > 1:
                    obj_std = [[stdev([matrix[i][j] for matrix in seeded_list_obj]) for j in range(len(seeded_list_obj[0][0]))] for i in range(len(seeded_list_obj[0]))]
                    df_obj_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([value for this_round in obj_std for value in this_round])
                    df_vendi_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]]  = str([stdev(i) for i in zip(*seeded_list_vendi)])
                else:
                    obj_std = [[0 for j in range(len(seeded_list_obj[0][0]))] for i in range(len(seeded_list_obj[0]))]
                    df_obj_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]] = str([value for this_round in obj_std for value in this_round])
                    df_vendi_stdev.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]]  = str([0 for i in zip(*seeded_list_vendi)])  

                df_vendi_av.loc[batch_names[batches.index(batch)],Vendi_names[Vendi_pruning_fractions.index(Vpf)]]  = str([sum(i) / len(i) for i in zip(*seeded_list_vendi)])



        # Save the dataframes with the processed results as csv files.        
        df_obj_av.to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{'__'.join(objectives)}]_av.csv"),index=True,header=True)
        df_obj_stdev.to_csv(wdir.joinpath(name_results+f"/benchmark_obj[{'__'.join(objectives)}]_stdev.csv"),index=True,header=True)
        df_vendi_av.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index=True,header=True)
        df_vendi_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_stdev.csv"),index=True,header=True)

        print(f"Data collection finished! Results are saved in the subfolder {name_results}.")


    def heatmap_plot(self,type_results, name_results, budget, scope_method = "geometric_mean",objective_mode = {"all_obj":"max"}, objective_weights=None,
                     bounds = {"rate":(2.349,1.035),"vendi":(6.366,1.941)},filename=None, show_plot=True,directory = '.'):
        """
        Generates and saves a heatmap plot for the requested result type across different batch sizes and Vendi_pruning_fractions.
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
                method to calculate the scope score ("average","product","geometric_mean" (Default))
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
        # Instantiate variable.
        df_obj = None
        dict_dfs_obj = None
        df_vendi = None
        df_heatmap = None

        # Prepare the DataFrame according to input.
        if ((type_results.lower() == "vendi") or (type_results.lower() == "scope")):
            csv_filename_results = wdir.joinpath(name_results+"/benchmark_vendi_av.csv")
            df_vendi = pd.read_csv(f"{csv_filename_results}",index_col=0,header=0, float_precision = "round_trip")  # contains the Vendi scores for each round in each element
            df_vendi = df_vendi.applymap(lambda x: ast.literal_eval(x))
            df_vendi = df_vendi.applymap(lambda x: x[-1])  # only keep the final Vendi score
            df_vendi = df_vendi.apply(pd.to_numeric)

        if type_results.lower() != "vendi":  # objective, specific objective, or scope score requested
            # get the objective names
            objectives = self.find_objectives(name_results)
            # check if a specific objective was requested
            if (type_results.lower() != "scope") and (type_results != "objective"):
                    objectives = [type_results]  # reassign the objectives list as the requested objective
            # read and process the objective data for each requested objective
            dict_dfs_obj = {}
            for objective in objectives:
                csv_filename_results = wdir.joinpath(name_results+f"/benchmark_obj[{objective}]_av.csv")
                dict_dfs_obj[objective] = pd.read_csv(f"{csv_filename_results}",index_col=0,header=0, float_precision = "round_trip")
                dict_dfs_obj[objective] = dict_dfs_obj[objective].applymap(lambda x: ast.literal_eval(x))  # contains the average objective values for each round in each element
                for batch in dict_dfs_obj[objective].index:
                    for column in dict_dfs_obj[objective].columns:
                        av_obj_list = dict_dfs_obj[objective].loc[batch,column]
                        batch_sizes = None
                        if type(batch) is not str:  # case using a fixed batch size
                            batch_sizes = [batch]*len(av_obj_list)
                        else:  # case using different batch sizes in each round
                            batch_sizes = [int(el) for el in batch.split("-")]  # list with the batch sizes for each round
                        if sum(batch_sizes) > budget:
                            difference = budget - sum(batch_sizes)
                            batch_sizes[-1] += difference  # reduce the last batch if it was smaller due to budget constraints
                        # reassign with the average obj value for the run
                        total_obj = [i*j for i,j in zip(batch_sizes,av_obj_list)]
                        dict_dfs_obj[objective].loc[batch,column] = sum(total_obj)/sum(batch_sizes)

                dict_dfs_obj[objective] = dict_dfs_obj[objective].apply(pd.to_numeric)
            
            # If there are more than one objectives or if the type_results="scope", the objectives need to be normalized
            if (len(dict_dfs_obj) > 1) or (type_results.lower() == "scope"): 
                for obj in objectives:
                    dict_dfs_obj[obj] = dict_dfs_obj[obj].applymap(lambda x: self.normalization(score=x,type="obj",obj_bounds=bounds[obj]))
                # check for minimization objectives
                min_obj = [obj for obj,value in objective_mode.items() if value == "min"]
                if min_obj:
                    for obj in min_obj:
                        dict_dfs_obj[obj] = dict_dfs_obj[obj].applymap(lambda x: 1-x)
            # average the objectives (or apply weights if provided)
            if (objective_weights is not None) and (len(objectives) > 1):
                # ensure that the weights sum up to 1
                sum_weights = sum(objective_weights.values())
                objective_weights = {obj: value/sum_weights for obj,value in objective_weights.items()}
            else:
                objective_weights = {obj: 1/len(objectives) for obj in objectives}
            # combine the objectives
            df_obj = sum(objective_weights[obj] * dict_dfs_obj[obj] for obj in dict_dfs_obj.keys())

        if type_results.lower() == "vendi":
            df_heatmap = df_vendi
        elif type_results.lower() == "scope":    
            # Scaling the Vendi data for the scope score calculation (objective data was already normalized above)
            df_vendi = df_vendi.applymap(lambda x: self.normalization(score=x,type="vendi",vendi_bounds=bounds["vendi"]))
            # calculate the scope score
            df_heatmap = self.calculate_scope_score(df_obj,df_vendi,scope_method)
        else:
            df_heatmap = df_obj

        if show_plot:
            # Generate and save the heatmap plot.
            plt.figure(figsize=(10,3))
            heatmap = sns.heatmap(df_heatmap,annot=True, fmt=".3f", linewidths=1,cmap='crest',cbar_kws={'label': f"{type_results} score"})
            heatmap.set(xlabel="Vendi pruning fraction in %", ylabel="batch size")
            heatmap.tick_params(length=0)
            plt.show()
            if filename is not None:
                heatmap_figure = heatmap.get_figure()
                heatmap_figure.savefig(wdir.joinpath(filename))
        return df_heatmap


    def progress_plot(self,budget,type_results, name_results, scope_method= "geometric_mean", objective_mode = {"all_obj":"max"},
                      objective_weights=None, bounds = {"rate":(2.349,1.035),"vendi":(6.366,1.941)},filename_figure = None, 
                      directory=".",show_plot=True):
        """
        Generates a result(number of experimenst) y(x)-plot for the requested results.
        Options for displayed results: scope score ("scope"), vendi score ("vendi"), weighted objectives ("objectives", normalized if multiple objectives),
        or individual objectives displayed by their respective name.

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
                method to calculate the scope score ("average","product","geometric_mean" (Default))
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
        """

        wdir = Path(directory)

        # Instantiate some variables.
        dict_raw_data = {}
        dict_unscaled_data = {}
        objectives = None
        unscaled_data = None
        index_list = None
        batch_sizes_list = None

        # Read in the required data
        if ((type_results.lower() == "vendi") or (type_results.lower() == "scope")):
            dict_raw_data["vendi"] = pd.read_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index_col=0,header=0, float_precision = "round_trip")  # contains the Vendi scores for each round in each element
        if type_results.lower() != "vendi":  # objective, specific objective, or scope score requested
            # get the objective names
            objectives = self.find_objectives(name_results)
            # check if a specific objective was requested
            if (type_results.lower() != "scope") and (type_results.lower() != "objective"):
                    objectives = [type_results]  # reassign the objectives list as the requested objective
            # read and process the objective data for each requested objective
            for objective in objectives:
                csv_filename_results = wdir.joinpath(name_results+f"/benchmark_obj[{objective}]_av.csv")
                dict_raw_data[objective] = pd.read_csv(f"{csv_filename_results}",index_col=0,header=0, float_precision = "round_trip")
        # Change data to processable format
        for key in dict_raw_data.keys():
            dict_raw_data[key] = dict_raw_data[key].applymap(lambda x: ast.literal_eval(x))

        # loop through all the dataframes to process them
        for key in dict_raw_data.keys():
            index_list = []
            batch_sizes_list = []
            results_list = []
            for batch in dict_raw_data[key].index:
                for Vpf in dict_raw_data[key].columns:
                    index_string = "b"+str(batch)+"_V"+str(Vpf)
                    index_list.append(index_string)
                    batch_sizes = None
                    if type(batch) is int:  # case using a fixed batch size
                        batch_sizes = [int(batch)]*len(dict_raw_data[key].loc[batch,Vpf])
                        difference = budget - sum(batch_sizes)
                        batch_sizes[-1] += difference  # reduce the last batch if it was smaller due to budget constraints
                    else:  # case using different batch sizes in each round
                        batch_sizes = [int(el) for el in batch.split("-")]  # list with the batch sizes for each round
                    batch_sizes_list.append(batch_sizes)
                    
                    processed_data = []  # list to store the processed data
                    if key == "vendi":
                        processed_data = dict_raw_data[key].loc[batch,Vpf]
                    else:  # processing of objective data
                        unprocessed_obj = dict_raw_data[key].loc[batch,Vpf]  # these are the average results per round, not for all experiments until this round!
                        total_obj = [i*j for i,j in zip(batch_sizes,unprocessed_obj)]  # [batch size]*[average obj] for each round
                        for round in range(len(unprocessed_obj)):
                            processed_result = sum(total_obj[:(round+1)]) / sum(batch_sizes[:(round+1)])
                            processed_data.append(processed_result)
                    results_list.append(processed_data)
            dict_unscaled_data[key] = pd.DataFrame(results_list,index_list).T

        # normalize the results if required
        # For type_results = objective, only normalize if there are multiple objectives (so that they can be properly weighted)
        # If there is only one objectives, the results are presented in natural values.
        if (type_results.lower() == "scope") or ((type_results.lower() == "objective") and (len(objectives) > 1)):
            for key in dict_unscaled_data.keys():
                dict_unscaled_data[key] = dict_unscaled_data[key].applymap(lambda x: self.normalization(score=x,obj_bounds=bounds[key]))
            # check for minimization objectives
            min_obj = [obj for obj, value in objective_mode.items() if value == "min"]
            if min_obj:
                for obj in min_obj:
                    dict_unscaled_data[obj] = dict_unscaled_data[obj].applymap(lambda x: 1-x)
        
        # average the objectives (or apply weights if provided; only required if type_results is not "vendi")
        if type_results.lower() != "vendi":
            if (objective_weights is not None) and (len(objectives) > 1):
                # ensure that the weights sum up to 1
                sum_weights = sum(objective_weights.values())
                objective_weights = {obj: value/sum_weights for obj,value in objective_weights.items()}
            else:
                objective_weights = {obj: 1/len(objectives) for obj in objectives}
            # combine the objectives
            dict_unscaled_data["comb_obj"] = sum(objective_weights[obj] * dict_unscaled_data[obj] for obj in objectives)
        
        # Assign the data that is displayed in the plot
        if type_results.lower() == "scope":
            unscaled_data = self.calculate_scope_score(dict_unscaled_data["comb_obj"],dict_unscaled_data["vendi"],scope_method)
        elif type_results.lower() == "vendi":
            unscaled_data = dict_unscaled_data["vendi"]
        else:
            unscaled_data = dict_unscaled_data["comb_obj"]

        #  The data is currently shown per batch (which can be different for the different runs). Still needs to be "scaled" to the number of experiments. 
        scaled_data = pd.DataFrame(np.nan,[x+1 for x in range(budget)],unscaled_data.columns)  # create empty dataframe with the right shape
        for column_nr in range(len(unscaled_data.columns)):
            batch_sizes = batch_sizes_list[column_nr]  # batch sizes for each round
            nr_experiments = [sum(batch_sizes[:round+1]) for round in range(len(batch_sizes))]  # total number of experiments including for each round
            for entry in range(len(unscaled_data.index)):
                if pd.notna(unscaled_data.iloc[entry,column_nr]):  # check that the value is actually numeric and not nan
                    scaled_data.iloc[nr_experiments[entry]-1, column_nr] = unscaled_data.iloc[entry,column_nr]  # -1 because iloc is 0-indexed

        # Plot and save the figure if requested.
        if show_plot:
            plt.figure(figsize=(10,10))
            plot = sns.lineplot(data=scaled_data)
            plt.xlabel('Number of selected samples')
            plt.ylabel(f"{type_results} score")
            plt.show()
            if filename_figure is not None:
                figure = plot.get_figure()
                figure.savefig(wdir.joinpath(filename_figure))


        return scaled_data.dropna(how="all")  # return the data


    def track_samples(self,filename_umap, filename_data,name_results,scope_method="geometric_mean", 
                      objective_mode = {"all_obj":"max"}, objective_weights=None, 
                      bounds = {"rate":(2.349,1.035),"vendi":(6.366,1.941)},display_cut_samples=True, obj_to_display = None, 
                      rounds_to_display = None, label_round=False,filename_figure=None,restrict_samples=None,directory='.'):
        """
        Visually tracks the evaluated and cut samples of a single benchmarking run on a provided UMAP.
        Saves the generated plot. Also provides the results for the run.
        ------------------------------------------------------------------------------------------------
        Inputs:
            filename_umap: str
                name of the file containing the UMAP coordinates
            filename_data: str
                name of the benchmarking run to be analyzed
            filename_figure: str or None
                name for the figure that is generated
                Default: None --> the figure is not saved
            name_results: str
                subfolder in which the results are located
            scope_method: str
                method to calculate the scope score ("average","product","geometric_mean" (Default))
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
                Default values are for the ArI dataset.
            display_cut_samples: Boolean
                show the samples that have been cut by the vendi scoring or not. Default = True.
                If True, the samples in the plot will be colored by the round when they were selected.
                If False, the samples will be displayed by their objective value 
                (default is first listed objective - can be changed in variable obj_to_display)
            obj_to_display: dict or None
                color the selected points by objective values if display_cut_samples is False.
                Default is None (take the first listed objective and its extreme values as bounds)
                Can also provide a dict with the objective name and its extreme values (max,min).
                E. g. : obj_to_display = {"yield":(100,0)}
            rounds_to_display: int or None
                Specify how many rounds of the run you want to display (starting from the first one).
                The metrics will also only be calculate for the these rounds.
                E. g.: rounds_to_display=5 --> first 4 rounds will be displayed
                Default is None --> shows all rounds
            label_round: Boolean
                label the suggested samples by the round of selection. Default = False.
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
            dict_obj_scaled[obj] = [self.normalization(score=value,type="obj",obj_bounds=bounds[obj]) for value in dict_obj_values[obj]]
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
        vendi_scaled = self.normalization(score=vendi_score,type="vendi")
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
        obj_plot_data = None
        obj_plot_bounds = None
        if obj_to_display is None:
            obj_plot_name = objectives[0]
            obj_plot = dict_obj_values[obj_plot_name]
            obj_plot_bounds = (max(obj_plot),min(obj_plot))
        else:
            obj_plot_name = next(iter(obj_to_display))
            obj_plot = dict_obj_values[obj_plot_name]
            obj_plot_bounds = obj_to_display[obj_plot_name]

        df_umap[obj_plot_name] = "PENDING"
        # Assign the samples to the df_umap dataframe.
        for round in list(df_data.index):            
            for sample in df_data.loc[round,"eval_samples"]:
                sample = sample.encode().decode('unicode_escape')
                df_umap.loc[sample,"status"] = "suggested"
                df_umap.loc[sample,"round"] = round+1
                df_umap.loc[sample,obj_plot_name] = obj_plot.pop(0) # this works because samples and objective values are in the same order

            if display_cut_samples:
                for sample in df_data.loc[round,"cut_samples"]:
                    sample = sample.encode().decode('unicode_escape')
                    df_umap.loc[sample,"status"] = "removed"
                    df_umap.loc[sample,"round"] = round+1 

        # Sort df_umap so that important points will be plotted last and won't be covered by neutral points.
        df_umap.sort_values("status",inplace=True,ascending=True)

        # Plot the results
        plt.figure(figsize=(10,8))

        if display_cut_samples:  # color by round
            colormap = 'crest'
            # Plot the non-selected points first
            plot = sns.scatterplot(
                data=df_umap[df_umap['round'] == 0], x="UMAP1", y="UMAP2", s=40,
                color='silver', style="status", marker="o",legend=False, alpha = 0.6,
                style_order=["suggested", "removed", "neutral"])

            # Plot the cut points
            plot = sns.scatterplot(
                data=df_umap[df_umap["status"] == "removed"], x="UMAP1", y="UMAP2", s=50,
                hue="round", palette=colormap, style="status", legend=False, alpha=0.7,
                style_order=["suggested", "removed", "neutral"], zorder=2)

            # Plot the selected points
            plot = sns.scatterplot(
                data=df_umap[df_umap["status"] == "suggested"],
                x="UMAP1", y="UMAP2", s=100, hue="round", palette=colormap,
                style="status", legend=False, edgecolor="k",linewidth=1,
                style_order=["suggested", "removed", "neutral"], zorder=3)

            # Add a colorbar for the 'hue' (selected/ removed points)
            norm = mpl.colors.Normalize(vmin=1, vmax=len(df_data.index))  # Normalize the colorscale
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])  # Empty array for ScalarMappable
            cbar = plt.colorbar(sm)
            cbar.set_label('Round')

        else:  # color by obj values
            # Separate selected and non-selected points
            df_selected = df_umap[pd.to_numeric(df_umap[obj_plot_name], errors='coerce').notna()].copy()
            df_selected[obj_plot_name] = df_selected[obj_plot_name].astype(float)
            df_pending = df_umap[pd.to_numeric(df_umap[obj_plot_name], errors='coerce').isna()]

            # Define colormap and normalization
            vmin = obj_plot_bounds[1]
            vmax = obj_plot_bounds[0]
            norm = plt.Normalize(vmin, vmax)
            cmap = sns.color_palette("vlag", as_cmap=True)

            # Plot non-numeric entries ("PENDING")
            plt.scatter(df_pending["UMAP1"], df_pending["UMAP2"], color="silver", s=20, alpha=0.6)

            # Plot numeric entries
            scatter_numeric = plt.scatter(df_selected["UMAP1"],df_selected["UMAP2"],c=df_selected[obj_plot_name],cmap=cmap,norm=norm,s=100,alpha=1,edgecolor='k',linewidth=1)

            # Add colorbar
            cbar = plt.colorbar(scatter_numeric)
            cbar.set_label(f"Average {obj_plot_name}")

        if label_round:  # label the round of selection if requested
            texts = []
            for i in df_umap[df_umap["status"]== "suggested"].index:
                texts.append(plt.text(df_umap.loc[i,'UMAP1'], df_umap.loc[i,'UMAP2'], int(df_umap.loc[i,'round']), size='medium', color='black', weight='semibold'))
            adjust_text(texts,expand_points=(1.3,1.3),force_static=(10,10),arrowprops={"arrowstyle":"-","color":"black"})
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.show()

        if filename_figure is not None:
            figure = plot.get_figure()
            
            figure.savefig(wdir.joinpath(filename_figure))
            
    
    def show_scope(self,filename_data,name_results,by_round=True,rounds_to_display=None,common_core=None,directory='.',molsPerRow=6):
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
            common_core: str
                string for the common core of the molecules to align them
                Default: None
            directory: str
                current directory. Default: "."
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
        
        def generate_representation(smiles_list):
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
        
        if by_round:
            for round in df_data.index:
                smiles_list = df_data.loc[round,"eval_samples"]
                mol_list = generate_representation(smiles_list)
                # Draw the aligned molecules
                print(f"Molecules selected in round {round+1}:")
                depiction = Draw.MolsToGridImage(
                    mol_list,
                    molsPerRow=len(mol_list),
                    subImgSize=(200, 200),
                    legends = [str(sample_dict[smiles]) for smiles in smiles_list]
                    )
                display(depiction)

        else:
            smiles_list = [smiles for round_list in [df_data.loc[round,"eval_samples"] for round in df_data.index] for smiles in round_list]
            mol_list = generate_representation(smiles_list)
            depiction = Draw.MolsToGridImage(
                mol_list,
                molsPerRow=molsPerRow,
                subImgSize=(200, 200),
                legends = [str(sample_dict[smiles]) for smiles in smiles_list]
                )
            display(depiction)


    @staticmethod
    def feature_analysis(filename,filename_labelled,objectives,objective_mode,
                           filename_shap="df_shap.csv",plot_type=["bar"],directory="."):
        
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
            objective_mode: list
                list of the mode of the objectives (max or min). E.g.: ["max","max"]
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


    @staticmethod
    def normalization(score,type="obj",vendi_bounds=(6.366,1941),obj_bounds=(2.349,1.035)):
        """
        Function to normalize Vendi scores and average objectives.
        Input:
            score: float or int
                result to be processed
            type: str
                type of results to be processed.
                Options:
                    "vendi": Vendi scores
                    "obj": average objectives (Default)
            vendi_bounds: tuple
                upper and lower bounds (in that order) of the vendi scores for the normalization
                Default values are for the ArI dataset.
            obj_bounds: tuple
                upper and lower bounds (in that order) of the average objectives for the normalization
                Default values are for the ArI dataset.
        """
        
        if "obj" in type.lower():
            return (score-obj_bounds[1])/(obj_bounds[0]-obj_bounds[1])
        elif type.lower() == "vendi":
            return (score-vendi_bounds[1])/(vendi_bounds[0]-vendi_bounds[1])
        else:
            return print("No valid type provided for the normalization!")
        
    @staticmethod
    def standardization(score,distr_metrics):
        """
        Standardizes a provided score.
        distr_metric: tuple
            (mean, stdev) --> both type float
        """
        return (score-distr_metrics[0])/distr_metrics[1]
    

    @staticmethod
    def calculate_scope_score(obj_score,vendi_score,method):
        """
        Calculate the scope score using different calculation methods.
        obj_score, vendi_score: float of the respective scaled score.
        method: string (Options "average", "product","geometric_mean")
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
          

    def results_for_run_conti(self,budget,type_results, name_results, scope_method="geometric_mean",scale_to_exp=True, directory="."):
        """
        Adapted version of progress_plot() to get the progress of the different metrics for a continue_data_collection() run.
        NOTE: Only works if there is just one run (can have multiple seeds) in the name_results folder.
        -------------------------
        budget: int
            scope size
        type_results: str
            type of results to be displayed
            options: "scope" (scope score), "vendi" (Vendi score), "objective" (objective score)
        name_results: str
            folder with results to be analyzed
        scope_method: str
            method to calculate the scope score ("average","product","geometric_mean" (Default))
        scale_to_exp: Boolean
            option to display the results by round (False) or number of experiments (True; Default)
        directory: str
            working directory. Default is current directory (".")
        """

        wdir = Path(directory)
            
        # Read in the required data.
        if (("obj" in type_results.lower()) or ("scope" in type_results.lower())):
            data = pd.read_csv(wdir.joinpath(name_results+"/benchmark_obj[rate]_av.csv"),index_col=0,header=0)
            data = data.applymap(lambda x: ast.literal_eval(x))
            if "scope" in type_results.lower():
                data2 = pd.read_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index_col=0,header=0, float_precision = "round_trip")
                data2 = data2.applymap(lambda x: ast.literal_eval(x))
        if "vendi" in type_results.lower():
            data = pd.read_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index_col=0,header=0, float_precision = "round_trip")
            data = data.applymap(lambda x: ast.literal_eval(x))
        
        index_list = []
        results_list = []
        results_list2 = []
        batch_sizes_list = []

        # Go through the results and analyze them.
        for batch in data.index:
            for Vendi_pruning_fraction in data.columns:
                index_string = "b"+str(batch)+"_V"+str(Vendi_pruning_fraction)
                index_list.append(index_string)

                # define the batch sizes
                batch_sizes = [batch]*len(data.loc[batch,Vendi_pruning_fraction])
                # redefine the first batch (these are the prior samples) based on the name of the individual runs
                batch_sizes[0] = int(os.listdir(wdir.joinpath(name_results+"/raw_data/"))[0].split("+")[0])
                difference = budget - sum(batch_sizes)
                batch_sizes[-1] += difference  # reduce the last batch if it was smaller due to budget constraints
                batch_sizes_list.append(batch_sizes)

                processed_data = []
                processed_data2 = []

                if (("obj" in type_results.lower()) or ("scope" in type_results.lower())):
                    unprocessed_obj = data.loc[batch,Vendi_pruning_fraction]  # these are the average yields per round, not for all experiments until this round!
                    processed_data = []
                    processed_data2 = []
                    total_obj = [i*j for i,j in zip(batch_sizes,unprocessed_obj)]  # [batch size]*[average obj] for each round
                    for round in range(len(unprocessed_obj)):
                        processed_result = sum(total_obj[:(round+1)]) / sum(batch_sizes[:(round+1)])
                        processed_data.append(processed_result)

                    if "scope" in type_results.lower():
                        processed_data2 = data2.loc[batch,Vendi_pruning_fraction]

                elif "vendi" in type_results.lower():
                    processed_data = data.loc[batch,Vendi_pruning_fraction]
                results_list.append(processed_data)
                results_list2.append(processed_data2)

        unscaled_data = pd.DataFrame(results_list,index_list).T
        unscaled_data2 = None

        if "scope" in type_results.lower():
            unscaled_data2 = pd.DataFrame(results_list2,index_list).T

            # Scale the dataframes for the calculation of the scope score (using the experimentally determined bounds).
            unscaled_data2 = unscaled_data2.applymap(lambda x: self.normalization(score=x,type="vendi"))
            unscaled_data = unscaled_data.applymap(lambda x: self.normalization(score=x,type="obj"))
            unscaled_data = self.calculate_scope_score(unscaled_data,unscaled_data2,scope_method)

        if scale_to_exp:
            #  The data is currently shown per batch (which can be different for the different runs). Still needs to be "scaled" to the number of experiments. 
            scaled_data = pd.DataFrame(np.nan,[x+1 for x in range(budget)],unscaled_data.columns)  # create empty dataframe with the right shape
            for column_nr in range(len(unscaled_data.columns)):
                batch_sizes = batch_sizes_list[column_nr]  # batch sizes for each round
                nr_experiments = [sum(batch_sizes[:round+1]) for round in range(len(batch_sizes))]  # total number of experiments including for each round
                for entry in range(len(unscaled_data.index)):
                    scaled_data.iloc[nr_experiments[entry]-1, column_nr] = unscaled_data.iloc[entry,column_nr]  # -1 because iloc is 0-indexed

            return scaled_data  # return the data
        
        else:
            prior_rounds = 10-int(name_results.split("/")[-1][5])
            recorded_rounds = len(unscaled_data.index)
            unscaled_data.index = range(prior_rounds,prior_rounds+recorded_rounds)
            return unscaled_data
    
    
    @staticmethod
    def find_objectives(folder_name):

        """
        Find the names of the objectives in the raw results.
        -------------------
        Input:
            folder_name: path of the folder to be analzyed
        -------------------
        Returns:
            list of the objectives that were used in this run
        """
        wdir = Path(".")
        # Get a list of the raw files of the run
        raw_path = wdir.joinpath(folder_name+"/raw_data/")
        raw_files = os.listdir(raw_path)
        # Load one of the files
        df_results = pd.read_csv(wdir.joinpath(raw_path,raw_files[0]),index_col=0,header=0)
        
        # Get the list of objectives from the column name and return it
        return ast.literal_eval(df_results.columns[0][11:])


    
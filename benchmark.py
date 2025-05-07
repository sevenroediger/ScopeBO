import ast
import os
from pathlib import Path
from statistics import stdev
import sys
import warnings

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import seaborn as sns

from scripts  import ScopeBO


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Benchmark:

    def __init__(self):
        pass

    
    def collect_data(self, filename_labelled, objectives, objective_mode, budget, batches, 
            Vendi_pruning_fractions, seeds, name_results, init_sampling_method, sample_threshold = None, enforce_dissimilarity=False, pruning_metric = "vendi", acquisition_function_mode = 'balanced', 
            dft_filename = None, filename_prediction = "df_benchmark.csv", directory='.'):
        """
        Runs the function for selected parameter ranges and records the results per round for 
        each set of parameters.

        Returns a number of csv files in a folder with the name name_results:
            Raw results:
                One per combination of seed, batch, and Vendi_pruning_fraction settings containing the following results:
                objective values for each evaluated point, grouped by round ("obj_value")
                Vendi score of the all evaluated points after each round ("Vendi_score")
                indices of evaluated samples for each round ("eval_samples")
                indices of samples cut by the Vendi cutoff for each round ("cut_samples")

                Example output (batch size 3 and 2 objectives; 4 samples were pruned from the reaction space):

                        objective_values        Vendi_score     samples     cut_samples
                0       [[1,2],[1,5],[5,4]]     43              [3,5,12]    [5,3,1,35,5]
                1       [[11,2],[11,5],[6,4]]   37              [5,17,1]    [4,9,7,24,51]

                Whereas the index column corresponds to the round of experiments.

                The raw result files are saved as [budget][acquisition_function_mode]_s[seed]_b[batch]_V[Vendi_pruning_fraction].csv
                They are located in a subfolder "raw_data".
            Summarized data:
                These files contain either objective value data (averaged for each batch round) ("obj") or
                Vendi score data ("vendi"). The rows correspond to the different batch sizes in batches (noted as indices). The columns
                correspond to the different Vendi_pruning_fractions.
                The entries are lists containing the results per round. There are files contain average values ("average") (across the different seeds) or 
                the standard deviation ("stdev") (across the different seeds).

                Example name: "benchmark_obj_av.csv" --> average objective value data
        
        -------------------------------------------------------------------------------------

        filename_labelled: string
            name of the csv file with the labelled data
        filename_prediction: string
            name of the csv file in which the benchmark dataframe is saved
            Default: "df_benchmark.csv"
        objective: list
            list indicating the column name for the objective (string) NOTE: Only a single objective is supported at the moment.
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

        # Generate a copy of the DataFrame with labelled data and remove the objective data. NOTE: only implemented for single objective benchmarks.
        df_unlabelled = df_labelled.copy(deep=True)
        df_unlabelled.drop(columns=[objectives[0]],inplace=True)

        # Instantiate empty for the analyzed results.
        Vendi_names = [str(x) for x in Vendi_pruning_fractions]
        batch_names = [str(x) for x in batches]
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
                            rounds = int(budget/batch)+1 # extra round with reduced batch size for last run (will be reduced below)
                        
                        else:
                            rounds = int(budget/batch)
                    
                    # Run ScopeBO for these settings.
                    for round in range(rounds):
                            
                        # check if the batch sie is dynamic (meaning different batch sizes for each rounds)
                        current_batch = None
                        if type(batch) is list:
                            current_batch = batch[round]
                        else:
                            current_batch = batch
                            # Check if this will be a run with reduced batch size (due to the set budget).
                            if round+1 == rounds and budget % batch != 0:
                                current_batch = budget % batch

                        # Check if the Vendi_pruning_fraction is dynamic (meaning different fractions for each round)
                        this_Vendi_pruning_fraction = None
                        if type(Vendi_pruning_fraction) is list:
                            this_Vendi_pruning_fraction = Vendi_pruning_fraction[round]
                        else:
                            this_Vendi_pruning_fraction = Vendi_pruning_fraction

                        print(f"Now running Batch size: {batch}, Vendi_pruning_fraction: {this_Vendi_pruning_fraction}, Seed: {seed}, Round: {round}, current batch: {current_batch}")
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

                        # Calculate the Vendi score for all points that were obseved so far.
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
                        for i in range(round):  # loop through the previously saved batches.
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
                    df_results = pd.DataFrame(raw_results,columns=["obj_values","Vendi_score","eval_samples","cut_samples"])
                    csv_filename_results = wdir.joinpath(name_results+f"/raw_data/{budget}{acquisition_function_mode}_s{seed}_b{batch}_V{Vendi_pruning_fraction}.csv")

                    df_results.to_csv(csv_filename_results,index=True,header=True)

                    print (f"Finished campaign {run_counter} of {total_runs}.")
                    run_counter+= 1
                
                # Calculate the averages and standard deviations across the differents seeds and save the results.
                df_obj_av.loc[str(batch),str(Vendi_pruning_fraction)]  = str([sum(i) / len(i) for i in zip(*seeded_list_obj)])
                # Standard deviation can only be calculated if there are more than 2 values. Set to zero if there is only one.
                if seeds > 1:
                    df_obj_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([stdev(i) for i in zip(*seeded_list_obj)])
                    df_vendi_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([stdev(i) for i in zip(*seeded_list_vendi)])
                else:
                    df_obj_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([0 for i in zip(*seeded_list_obj)])
                    df_vendi_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([0 for i in zip(*seeded_list_vendi)])  

                df_vendi_av.loc[str(batch),str(Vendi_pruning_fraction)]  = str([sum(i) / len(i) for i in zip(*seeded_list_vendi)])



        # Save the dataframes with the processed results as csv files.        
        df_obj_av.to_csv(wdir.joinpath(name_results+"/benchmark_obj_av.csv"),index=True,header=True)
        df_obj_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_obj_stdev.csv"),index=True,header=True)
        df_vendi_av.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index=True,header=True)
        df_vendi_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_stdev.csv"),index=True,header=True)

        print(f"Data collection finished! Results are saved in the subfolder {name_results}.")


    def continue_data_collection(self, filename_labelled, objectives, objective_mode, copy_rounds, remaining_budget, batches, 
            Vendi_pruning_fractions, seeds, name_results, name_prior_results, init_sampling_method, sample_threshold = None, enforce_dissimilarity=False, pruning_metric = "vendi", acquisition_function_mode = 'balanced', 
            dft_filename = None, filename_prediction = "df_benchmark.csv", directory='.'):
        """
        Takes the first n rounds from a different run and continues them with the indicated settings. Otherwise, the same as collect_data().
        See the docstring of collect_data() for full details on the generated reports and most variables.
        NOTE: The function only works with the folder name_prior_results only contains one run (with different seeds)!
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

        # Instantiate empty for the analyzed results.
        Vendi_names = [str(x) for x in Vendi_pruning_fractions]
        batch_names = [str(x) for x in batches]
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

                    # same for the pruned samples
                    prior_cut = [smiles.encode().decode('unicode_escape') for round_list in [df_prior_data.loc[round,"cut_samples"] for round in df_prior_data.index[:copy_rounds]] for smiles in round_list]

                    # typically there is no pruning in the first round of experiments (initiation), resulting in the list containing an empty element
                    if "" in prior_cut:
                        prior_cut.remove("")

                    # assign the priorities in the dataframe
                    current_df["priority"] = 0
                    current_df.loc[prior_selected,"priority"] = -2
                    current_df.loc[prior_cut,"priority"] = -1

                    # assign the objective values in the dataframe
                    current_df[objectives[0]] = "PENDING"
                    for idx in prior_selected:
                        current_df.loc[idx,"rate"] = df_labelled.loc[idx,"rate"]

                    current_df.to_csv(csv_filename_pred, index=True, header=True)
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
                        if remaining_budget % batch != 0:
                            rounds = int(remaining_budget/batch)+1 # extra round with reduced batch size for last run (will be reduced below)
                        
                        else:
                            rounds = int(remaining_budget/batch)
                    
                    # Run ScopeBO for these settings.
                    for round in range(rounds):
                            
                        # check if the batch sie is dynamic (meaning different batch sizes for each rounds)
                        current_batch = None
                        if type(batch) is list:
                            current_batch = batch[round]
                        else:
                            current_batch = batch
                            # Check if this will be a run with reduced batch size (due to the set budget).
                            if round+1 == rounds and _remaining_budget % batch != 0:
                                current_batch = remaining_budget % batch

                        # Check if the Vendi_pruning_fraction is dynamic (meaning different fractions for each round)
                        this_Vendi_pruning_fraction = None
                        if type(Vendi_pruning_fraction) is list:
                            this_Vendi_pruning_fraction = Vendi_pruning_fraction[round]
                        else:
                            this_Vendi_pruning_fraction = Vendi_pruning_fraction

                        print(f"Now running Batch size: {batch}, Vendi_pruning_fraction: {this_Vendi_pruning_fraction}, Seed: {seed}, Round: {round}, current batch: {current_batch}")
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

                        # Calculate the Vendi score for all points that were obseved so far.
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
                        for i in range(round):  # loop through the previously saved batches.
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
                    df_results = pd.DataFrame(raw_results,columns=["obj_values","Vendi_score","eval_samples","cut_samples"])
                    csv_filename_results = wdir.joinpath(name_results+f"/raw_data/{len(prior_selected)}+{remaining_budget}{acquisition_function_mode}_s{seed}_b{batch}_V{Vendi_pruning_fraction}.csv")

                    df_results.to_csv(csv_filename_results,index=True,header=True)

                    print (f"Finished campaign {run_counter} of {total_runs}.")
                    run_counter+= 1
                
                # Calculate the averages and standard deviations across the differents seeds and save the results.
                df_obj_av.loc[str(batch),str(Vendi_pruning_fraction)]  = str([sum(i) / len(i) for i in zip(*seeded_list_obj)])
                # Standard deviation can only be calculated if there are more than 2 values. Set to zero if there is only one.
                if seeds > 1:
                    df_obj_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([stdev(i) for i in zip(*seeded_list_obj)])
                    df_vendi_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([stdev(i) for i in zip(*seeded_list_vendi)])
                else:
                    df_obj_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([0 for i in zip(*seeded_list_obj)])
                    df_vendi_stdev.loc[str(batch),str(Vendi_pruning_fraction)]  = str([0 for i in zip(*seeded_list_vendi)])  

                df_vendi_av.loc[str(batch),str(Vendi_pruning_fraction)]  = str([sum(i) / len(i) for i in zip(*seeded_list_vendi)])



        # Save the dataframes with the processed results as csv files.        
        df_obj_av.to_csv(wdir.joinpath(name_results+"/benchmark_obj_av.csv"),index=True,header=True)
        df_obj_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_obj_stdev.csv"),index=True,header=True)
        df_vendi_av.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_av.csv"),index=True,header=True)
        df_vendi_stdev.to_csv(wdir.joinpath(name_results+"/benchmark_vendi_stdev.csv"),index=True,header=True)

        print(f"Data collection finished! Results are saved in the subfolder {name_results}.")


    def heatmap_plot(self,type_results, name_results, budget, filename=None, scaling="normalization", directory = '.'):
        """
        Generates and saves a heatmap plot for the requested result type across different batch sizes and Vendi_pruning_fractions.
        ---------------------------------------------------------------------------------------------------------
        Input:
            type_results: str
                Requested type of result.
                Options:
                    "vendi": Vendi score
                    "objective": objective values
                    "scope": scope score = vendi * objective
            name_results: str
                Name of the subfolder in which the result csv files are saved.
            budget: int
                experimental budget used in the runs
            filename: str or None
                name for the figure that is created. Default is None --> figure is not saved.
            directory: str
                current directory. Default: '.'
        """

        wdir = Path(directory)
        # Instantiate variable.
        df_obj = None
        df_vendi = None
        df_heatmap = None

        # Prepare the DataFrame according to input.
        if ((type_results.lower() == "vendi") or (type_results.lower() == "scope")):
            csv_filename_results = wdir.joinpath(name_results+"/benchmark_vendi_av.csv")
            df_vendi = pd.read_csv(f"{csv_filename_results}",index_col=0,header=0, float_precision = "round_trip")  # contains the Vendi scores for each round in each element
            df_vendi = df_vendi.applymap(lambda x: ast.literal_eval(x))
            df_vendi = df_vendi.applymap(lambda x: x[-1])  # only keep the final Vendi score
            df_vendi = df_vendi.apply(pd.to_numeric)

        if (("obj" in type_results.lower()) or (type_results.lower() == "scope")):
            csv_filename_results = wdir.joinpath(name_results+"/benchmark_obj_av.csv")
            df_obj = pd.read_csv(f"{csv_filename_results}",index_col=0,header=0, float_precision = "round_trip")
            df_obj = df_obj.applymap(lambda x: ast.literal_eval(x))  # contains the average objective values for each round in each element
            for batch in df_obj.index:
                for column in df_obj.columns:
                    av_obj_list = df_obj.loc[batch,column]
                    batch_sizes = None
                    if type(batch) is not str:  # case using a fixed batch size
                        batch_sizes = [batch]*len(av_obj_list)
                        difference = budget - sum(batch_sizes)
                        batch_sizes[-1] -= difference  # reduce the last batch if it was smaller due to budget constraints
                    else:  # case using different batch sizes in each round
                        batch_sizes = ast.literal_eval(batch)  # list with the batch sizes for each round
                    # reassign with the average obj value for the run
                    total_obj = [i*j for i,j in zip(batch_sizes,av_obj_list)]
                    df_obj.loc[batch,column] = sum(total_obj)/sum(batch_sizes)

            df_obj = df_obj.apply(pd.to_numeric)

        if type_results.lower() == "vendi":
            df_heatmap = df_vendi
        if type_results.lower() == "objective":
            df_heatmap = df_obj
        if type_results.lower() == "scope":    

            # Scaling the data for the scope score calculation.
            if scaling.lower() == "normalization":
                df_vendi = df_vendi.applymap(lambda x: self.normalization(score=x,type="vendi"))
                df_obj = df_obj.applymap(lambda x: self.normalization(score=x,type="obj"))
            elif scaling.lower() == "standardization":
                df_vendi = df_vendi.applymap(lambda x: self.standardization(score=x,type="vendi"))
                df_obj = df_obj.applymap(lambda x: self.standardization(score=x,type="obj"))
            else:
                return print("No valid scaling metric provided.")

            # The scope score is here defined as vendi * objective.
            df_heatmap = df_vendi * df_obj

        # Generate and save the heatmap plot.
        plt.figure(figsize=(10,3))
        heatmap = sns.heatmap(df_heatmap,annot=True, fmt=".3f", linewidths=1,cmap='crest',cbar_kws={'label': f"{type_results} score"})
        heatmap.set(xlabel="Vendi pruning fraction in %", ylabel="batch size")
        heatmap.tick_params(length=0)
        plt.show()
        if filename is not None:
            heatmap_figure = heatmap.get_figure()
            heatmap_figure.savefig(wdir.joinpath(filename))


    def progress_plot(self,budget,type_results, name_results,filename_figure = None, scaling="normalization", directory="."):
        """
        Generates a result(round) y(x)-plot for the requested results.

        Inputs:     #NOTE: fix docstring! Not up to date!        
            type_results: str
                Requested type of result.
                Options:
                    "vendi": Vendi score
                    "objective": objective values
                    "scope": scope score = vendi * objective
             batches: list
                list of batch sizes (as int numbers) for which the results will be displayed.
            Vendi_prunins: list
                list of threshold values for the Vendi cutoff (floats) for which the results will be displayed.
            filename: str
                name for the figure that is created
        """

        wdir = Path(directory)
            
        # Read in the required data.
        if (("obj" in type_results.lower()) or ("scope" in type_results.lower())):
            data = pd.read_csv(wdir.joinpath(name_results+"/benchmark_obj_av.csv"),index_col=0,header=0)
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
                index_string = "batch"+str(batch)+"_Vendi_pruning_fraction"+str(Vendi_pruning_fraction)
                index_list.append(index_string)

                batch_sizes = None
                if type(batch) is not str:  # case using a fixed batch size
                    batch_sizes = [batch]*len(data.loc[batch,Vendi_pruning_fraction])
                    difference = budget - sum(batch_sizes)
                    batch_sizes[-1] -= difference  # reduce the last batch if it was smaller due to budget constraints
                else:  # case using different batch sizes in each round
                    batch_sizes = ast.literal_eval(batch)  # list with the batch sizes for each round
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
            if scaling.lower() == "normalization":
                unscaled_data2 = unscaled_data2.applymap(lambda x: self.normalization(score=x,type="vendi"))
                unscaled_data = unscaled_data.applymap(lambda x: self.normalization(score=x,type="obj"))
            elif scaling.lower() == "standardization":
                unscaled_data2 = unscaled_data2.applymap(lambda x: self.standardization(score=x,type="vendi"))
                unscaled_data = unscaled_data.applymap(lambda x: self.standardization(score=x,type="obj"))
            else:
                return print("No valid scaling metric provided.")
            unscaled_data = unscaled_data * unscaled_data2

        #  The data is currently shown per batch (which can be different for the different runs). Still needs to be "scaled" to the number of experiments. 
        scaled_data = pd.DataFrame(np.nan,[x+1 for x in range(budget)],unscaled_data.columns)  # create empty dataframe with the right shape
        for column_nr in range(len(unscaled_data.columns)):
            batch_sizes = batch_sizes_list[column_nr]  # batch sizes for each round
            nr_experiments = [sum(batch_sizes[:round+1]) for round in range(len(batch_sizes))]  # total number of experiments including for each round
            for entry in range(len(unscaled_data.index)):
                scaled_data.iloc[nr_experiments[entry]-1, column_nr] = unscaled_data.iloc[entry,column_nr]  # -1 because iloc is 0-indexed

        # Plot and save the figure if requested.
        plt.figure(figsize=(10,10))
        plot = sns.lineplot(data=scaled_data)
        plt.xlabel('Number of selected samples')
        plt.ylabel(f"{type_results} score")
        plt.show()
        if filename_figure is not None:
            figure = plot.get_figure()
            figure.savefig(wdir.joinpath(filename_figure))


        return scaled_data.dropna(how="all")  # return the data


    def track_samples(self,filename_umap, filename_data,name_results,display_cut_samples=True,label_round=False,scaling="normalization",filename_figure=None,directory='.'):
        """
        Visually tracks the evaluated and cut samples of a single benchmarking run on a provided UMAP.
        Saves the generated plot.
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
            display_cut_samples: Boolean
                show the samples thta have been cut by the vendi scoring or not. Default = True.
            label_round: Boolean
                label the suggested samples by the round of selection. Default = False.
            directory: str
                current directory. Default: "."
        """

        # Set directory.
        wdir = Path(directory)

        # Read in UMAP and data.
        df_umap = pd.read_csv(wdir.joinpath(filename_umap),index_col=0, float_precision = "round_trip")
        df_data = pd.read_csv(wdir.joinpath(name_results+"/"+filename_data), float_precision = "round_trip")
        df_data["eval_samples"] = df_data["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
        df_data["cut_samples"] = df_data["cut_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
        df_data["obj_values"] = df_data["obj_values"].apply(ast.literal_eval)
        
        # Get the overall results for this run:
        vendi_score = df_data['Vendi_score'].iloc[-1]
        obj_scores = [value for round_list in [df_data.loc[round,"obj_values"] for round in df_data.index] for value in round_list]
        av_obj = sum(obj_scores)/len(obj_scores)
        vendi_scaled = self.normalization(score=vendi_score,type="vendi")
        obj_scaled = self.normalization(score=av_obj,type="obj")
        # Scaling the data for the scope score calculation.
        if scaling.lower() == "normalization":
            vendi_scaled = self.normalization(score=vendi_score,type="vendi")
            obj_scaled = self.normalization(score=av_obj,type="obj")
        elif scaling.lower() == "standardization":
            vendi_scaled = self.standardization(score=vendi_score,type="vendi")
            obj_scaled = self.standardization(score=av_obj,type="obj")
        else:
            return print("No valid scaling metric provided.")
        scope_score  =vendi_scaled * obj_scaled
        print(f"Scope score: {scope_score:.3f}; average objective: {av_obj:.3f}; Vendi score: {vendi_score:.3f}")

        # Add columns and reassign the index.
        df_umap["status"] = "neutral"
        df_umap["round"] = 0

        # Assign the samples to the df_umap dataframe. Also add a size parameter to make important samples bigger.
        df_umap["size"] = 10
        for round in list(df_data.index):            
            for sample in df_data.loc[round,"eval_samples"]:
                sample = sample.encode().decode('unicode_escape')
                df_umap.loc[sample,"status"] = "suggested"
                df_umap.loc[sample,"round"] = round+1 
                df_umap.loc[sample,"size"] = 15
            if display_cut_samples:
                for sample in df_data.loc[round,"cut_samples"]:
                    sample = sample.encode().decode('unicode_escape')
                    df_umap.loc[sample,"status"] = "removed"
                    df_umap.loc[sample,"round"] = round+1 
                    df_umap.loc[sample,"size"] = 15

        
        # Sort df_umap so that important points will be plotted last and won't be covered by neutral points.
        df_umap.sort_values("status",inplace=True,ascending=True)

        colormap = 'crest'
        # Create a mask for points with 'round' == 0 (these are the ones that were not selected)
        mask = df_umap['round'] == 0

        plt.figure(figsize=(8,6))
        # Plot the 'round' == 0 points (silver) first
        plot = sns.scatterplot(
            data=df_umap[mask],  # Only select rows where round == 0
            x="UMAP1", y="UMAP2",
            size="size", size_norm=(10, 20),
            color='lavender',  # Manually set the color to silver for these points
            style="status", legend=False,
            style_order=["suggested", "removed", "neutral"]
        )

        # Plot the other points
        plot = sns.scatterplot(
            data=df_umap[~mask],  # Select rows where round != 0
            x="UMAP1", y="UMAP2",
            size="size", size_norm=(10, 20),
            hue="round", palette=colormap,
            style="status", legend=False,
            style_order=["suggested", "removed", "neutral"],
            zorder=2  # Ensure these points are plotted on top
        )

        # Add a colorbar for the 'hue' (selected/ removed points)
        norm = mpl.colors.Normalize(vmin=1, vmax=len(df_data.index))  # Normalize the colorscale
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Empty array for ScalarMappable
        cbar = plt.colorbar(sm)
        cbar.set_label('Round')  # Label for the colorbar
        if label_round:
            for i in df_umap[df_umap["status"]== "suggested"].index:
                plt.text(df_umap.loc[i,'UMAP1'] + 0.1, df_umap.loc[i,'UMAP2'] + 0.1, int(df_umap.loc[i,'round']), 
                        horizontalalignment='left', 
                        size='small', color='black', weight='semibold')
        plt.show()

        if filename_figure is not None:
            figure = plot.get_figure()
            figure.savefig(wdir.joinpath(filename_figure))

    
    def show_scope(self,filename_data,name_results,by_round=True,common_core=None,directory='.',molsPerRow=6):
        """
        Depict the substrates that were selected for the scope.
        ------------------------------------------------------------------------------------------------
        Inputs:
            filename_data: str
                name of the benchmarking run to be analyzed
            name_results: str
                subfolder in which the results are located
            by_round: Boolean
                Select if the selected compounds are shown by round (True --> Default) or all together
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
        df_data["eval_samples"] = df_data["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
        df_data["obj_values"] = df_data["obj_values"].apply(ast.literal_eval)

        # Get a dictionary that maps the samples to the obj_values.
        sample_dict = {}
        for _, row in df_data.iterrows():
            labels = row["eval_samples"]
            values = row["obj_values"]
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
                print(f"Molecules selected in round {round}:")
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
    def feature_analysis(objectives,objective_mode,filename_data,filename_labelled_data,filename_analysis="feature_analysis.csv",directory="."):
        """
        Get the SHAP features from a benchmarking run using the ScopeBO function of the same name.

        # NOTE: complete docstring
        """

        wdir = Path(".")
        df_labelled = pd.read_csv(wdir.joinpath(filename_labelled_data),index_col=0,header=0)  # file with the labelled data
        df_data = pd.read_csv(wdir.joinpath(filename_data),index_col=0,header=0)  # result file of the benchmarking scope run

        # get the evaluated samples
        df_data["eval_samples"] = df_data["eval_samples"].apply(lambda x: [y.strip("'") for y in x[1:-1].split(', ')])
        idx_samples = [sample for round_list in [df_data.loc[round,"eval_samples"] for round in df_data.index] for sample in round_list]

        # put the objective values of all samples that were not evaluated as "PENDING"
        for objective in objectives:
            for idx in df_labelled.index:
                if idx not in idx_samples:
                    df_labelled.loc[idx,objective] = "PENDING"

        # save the file
        df_labelled.to_csv(wdir.joinpath(filename_analysis),index=True,header=True)

        # run the ScopeBO function
        shap_values, mean_abs_shap_values= ScopeBO().feature_analysis(objectives=objectives,objective_mode=objective_mode,filename=filename_analysis,directory=directory)
        return shap_values, mean_abs_shap_values


    @staticmethod
    def normalization(score,type):
        # Defining the bounds for the normalization.
        vendi_max = 6.333  # obtained with explorative acquisition function
        vendi_min = 2.036  # obtained with exploitative acq. fct.
        obj_max = 2.443  # obtained with exploitative acq. fct.
        obj_min = 0.969  #obtained with explorative acq. fct.
        
        if "obj" in type.lower():
            return (score-obj_min)/(obj_max-obj_min)
        elif type.lower() == "vendi":
            return (score-vendi_min)/(vendi_max-vendi_min)
        else:
            return print("No valid type provided for the normalization!")
        

    @staticmethod
    def standardization(score,type):
        # Bounds of standardization (obtained by random sampling 1 million times)
        vendi_mean = 2.61870733703201
        vendi_std = 0.19057210821261894
        obj_mean = 1.2723618238799999
        obj_std = 0.12407787948204162
        if "obj" in type.lower():
            return (score-obj_mean)/(obj_std)  # shifted to avoid negative values
        elif type.lower() == "vendi":
            return (score-vendi_mean)/(vendi_std)  # shifted to avoid negative values
        else:
            return print("No valid type provided for the normalization!")

    
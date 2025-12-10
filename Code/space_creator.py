import os
import pandas as pd
import numpy as np
from itertools import product as it_product
from pathlib import Path

def create_reaction_space(reactants,
                          feature_processing=True,
                          save_data = True,
                          directory='./',
                          filename='reaction_space.csv'):

    """
    Reaction scope generator
    Pass csv files with the different reaction reactants and their featurization. 
    Function returns a csv file with the reaction space.

    ------------------------------------------------------------------------------

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
    feature_processing: Boolean
        Option to preprocess the features. Default is True.
    save_data: Boolean
        Option to save the generated reaction space as a csv file. Default is True.
    directory: string
        set the working directory. Default is current directory.
    filename: string
        Filename of the output csv file. Default is reaction_space.csv
    """
    
    # Set working directory.
    wdir = Path(directory)
    # Assert that the type of reactants fits the requirements.
    msg="Please provide the reactants as a list or dictionary of filenames and submit again."
    assert type(reactants) == list or type(reactants) == dict, msg

    list_of_dfs = []
    i = 0
    # Generate DataFrames for each component and add them to list_of_dfs.
    for component in reactants:
        csv_component = wdir.joinpath(component)

        # Assertions for existence and type of file.
        msg = "The file " + component + " was not found. Please check your input and submit again."
        assert os.path.exists(csv_component), msg
        
        # Read and clean data.
        df_component_i = pd.read_csv(csv_component, float_precision = "round_trip")
        df_component_i = df_component_i.dropna(axis='columns', how='all')

        # Set data types for all columns.
        df_component_i.iloc[:,0] = df_component_i.iloc[:,0].astype(str)
        df_component_i.iloc[:,1:] = df_component_i.iloc[:,1:].astype(float)

        # Add labels for the column names according to the reaction component.
        column_names = df_component_i.columns.tolist()
        column_names = [str(name) for name in column_names]
        if type(reactants) is dict:
            column_names = [reactants[component]+"_"+name for name in column_names]
        elif len(reactants) > 1:
            column_names = ["reactant" + str(i+1)+"_"+name for name in column_names]
        df_component_i.columns = column_names
        # Add DataFrame to list.
        list_of_dfs.append(df_component_i)
        # Increase running variable for the component labelling.
        i += 1

    #Generate combinations of reactants for the reaction space.
    combinations = list(it_product(*(df.itertuples(index=False) for df in list_of_dfs)))
    # Convert to a DataFrame
    df_comb = pd.DataFrame([sum((list(row) for row in combination), []) for combination in combinations],
                      columns=sum((list(df.columns) for df in list_of_dfs), []))
    
    # Get the reactant combinations for each row.
    names_comb = []
    for _, row in df_comb.iterrows():
        # Only compound names are string values.
        names = row[row.apply(lambda x: isinstance(x, str))].tolist()
        combined_name = ""
        for i in range(len(names)):
            combined_name += names[i]
            if i+1 != len(names):
                combined_name += "."
        names_comb.append(combined_name)

    # Set the name of the reactant combination as the index. Delete columns with compound names.
    df_space = df_comb.loc[:, df_comb.apply(lambda col: not col.apply(lambda x: isinstance(x, str)).all())]
    df_space.index = names_comb

    # Preprocess the features if requested.
    if feature_processing:
        print("Now doing feature preprocessing.")
        df_space = feature_preprocessing(df_space)

    print("Generation of reaction space completed!")

    if save_data:
        csv_filename = wdir.joinpath(filename)  # sets name of output
        df_space.to_csv(csv_filename, index=True, mode = 'w', header=True)
        print(f"The search space has been saved in the file '{filename}.'")

    return df_space

def feature_preprocessing(df):
    """
    Function for removing non-varied and highly correlated features.
    Take a df as input and returns it in processed form.
    """
    # Remove columns that have only one unique value.
    removed_columns = []
    for column in df.columns:
        if len(np.unique(df[column].values)) <= 1:
            removed_columns.append(column)
    df = df.drop(removed_columns, axis=1)
    
    # Remove highly correlated features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df = df.drop(to_drop, axis=1)

    # Store the names of the column removed due to correlation
    for column in to_drop:
        removed_columns.append(column)

    print(f"The following features were removed: {removed_columns}")
    print(f"The final search space has {len(df.columns)} features.")
    
    return df

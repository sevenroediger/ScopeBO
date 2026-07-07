from pathlib import Path

from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.preprocessing import scale
from umap import UMAP

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

# Define a colormap for plotting
doyle_colors = ["#CE4C6F", "#1561C2", "#188F9D","#C4ADA2","#515798", "#CB7D85", "#A9A9A9"]
colors = [doyle_colors[1],"#FFFFFFD1",doyle_colors[0]]
cont_cmap = LinearSegmentedColormap.from_list("Doyle_cont", colors)


def UMAP_suggestions(filename,
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

    df_umap, df_scope, obj_name = _UMAP_creation(filename = filename,
                             obj_to_show = obj_to_show,
                             objectives = objectives,
                             directory = directory)
    
    # add the sample status to the df ("priority")
    if "priority" in df_scope.columns.to_list():
        df_umap["priority"] = df_scope["priority"]
    else:  # assume all samples are measured if no priority column
        df_umap["priority"] = -2

    # separate the labelled and unlabelled samples
    df_seen = df_umap.loc[df_umap[obj_name] != "PENDING"].copy()
    df_seen[obj_name] = df_seen[obj_name].astype(float)
    df_unseen = df_umap.loc[df_umap[obj_name] == "PENDING"]

    df_neutral = df_unseen[df_unseen["priority"] == 0].copy()
    df_cut = df_unseen[df_unseen["priority"] == -1].copy()
    df_sugg = df_unseen[df_unseen["priority"] == 1].copy()  # suggested samples
    mask = (df_unseen["priority"] > 0) & (df_unseen["priority"] < 1)
    df_alt = df_unseen[mask].copy()  # alternative suggestions

    # draw the structures if requested
    if draw_structures:
        # Extract the suggested molecules and draw them.
        scope_substrates  = df_seen.index.to_list()
        scope_substrates = [str(entry.encode().decode('unicode_escape')) for entry in scope_substrates]
        scope_labels = df_seen[obj_name].to_list()

        print("Structures of the scope substrates:")
        try:
            mol_list = [Chem.MolFromSmiles(smiles) for smiles in scope_substrates]
            # Draw the aligned molecules
            depiction = Draw.MolsToGridImage(
                mol_list,
                molsPerRow=6,
                subImgSize=(200, 200),
                legends=[str(result_val) for result_val in scope_labels]
                )
            display(depiction)
        except:
            print(f"Could not draw the molecules in {scope_substrates}.")
            print(f"Please label your molecules with SMILES strings to draw the results of the scope.")
            print("Alternatively, set draw_structures = False in the function call to avoid structure drawing.")

    # print the UMAP if requested
    if show_figure:
        print("UMAP projection of the reaction space:")
        if display_cut_samples:
            print("(Evaluated samples are shown as colored circles. Pruned samples are marked with X.)")
        else:
            print("(Evaluated samples are shown as colored circles.)")
        if display_suggestions and display_alternatives:
            print("(Suggested samples are marked with squares and alternative suggestions with diamonds.)")
        elif display_suggestions and not display_alternatives:
            print("(Suggested samples are marked with squares.)")
        elif not display_suggestions and display_alternatives:
            print("(Alternative suggestions are marked with diamonds.)")
        plt.figure(figsize=figsize, dpi = dpi, constrained_layout = True)

        colormap = cont_cmap
        if obj_bounds is None:
            vmin = df_seen[obj_name].min()
            vmax = df_seen[obj_name].max()
        else:
            vmin = obj_bounds[1]
            vmax = obj_bounds[0]
        norm = plt.Normalize(vmin,vmax)

        # plot the neutral points
        plt.scatter(
            df_neutral["UMAP1"], df_neutral["UMAP2"], s=40, 
            linewidth=0.3, edgecolor="k", color=doyle_colors[6],
            marker="o", alpha = 0.8, label = "neutral", zorder=1)

        # check if cut samples should be highlighted
        if display_cut_samples:
            # plot the cut samples
            plt.scatter(
                df_cut["UMAP1"], df_cut["UMAP2"], s=100, edgecolor="k", marker = "X",
                color=doyle_colors[4], alpha=0.6, label = "cut", linewidth = 0.3, zorder=2)
        else:
            # plot the cut samples the same way as the neutral samples
            plt.scatter(
                df_cut["UMAP1"], df_cut["UMAP2"], s=40, edgecolor="k", marker = "o",
                color=doyle_colors[6], alpha=0.8, linewidth = 0.3, zorder=2)
            
        # check if suggested samples should be highlighted
        if display_suggestions:
            # plot the suggested samples
            plt.scatter(
                df_sugg["UMAP1"], df_sugg["UMAP2"], s=100, edgecolor="k", marker = "s",
                color=doyle_colors[2], alpha=0.8, label = "suggested", linewidth = 1.2, zorder=4)
        else:
            # plot the sugggested samples the same way as the neutral samples
            plt.scatter(
                df_sugg["UMAP1"], df_sugg["UMAP2"], s=40, edgecolor="k", marker = "o",
                color=doyle_colors[6], alpha=0.8, linewidth = 0.3, zorder=1)
            
        # check if alternative sugegstions should be highlighted
        if display_alternatives:
            # plot the alternative samples
            plt.scatter(
                df_alt["UMAP1"], df_alt["UMAP2"], s=100*df_alt["priority"], edgecolor="k", marker = "D",
                color=doyle_colors[2], alpha=0.8, label = "alt. suggestion", linewidth = 1.2, zorder=3)
        else:
            # plot the alt. sugg. samples the same way as the neutral samples
            plt.scatter(
                df_alt["UMAP1"], df_alt["UMAP2"], s=40, edgecolor="k", marker = "o",
                color=doyle_colors[6], alpha=0.8, linewidth = 0.3, zorder=1)
        
        # plot the selected samples
        scatter_numeric = plt.scatter(df_seen["UMAP1"], df_seen["UMAP2"], c=df_seen[obj_name],
                                        cmap=cont_cmap, norm=norm,s=250, alpha=1, edgecolor='k', 
                                        label = "measured", linewidth=2, zorder = 5)

        cbar = plt.colorbar(scatter_numeric)
        if cbar_title is None:
            cbar_label = obj_name.capitalize()
            cbar.set_label(cbar_label)
        else:
            cbar.set_label(cbar_title)

        plt.legend()
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.show()

    if return_dfs:
        return {"seen": df_seen, 
                "suggested": df_sugg,
                "alternatives": df_alt,
                "neutral": df_neutral, 
                "cut": df_cut}
    

def UMAP_predictions(filename,
                      df_pred,
                      obj_to_show = None,
                      obj_bounds = None,
                      objectives = None,
                      pred_for_cut = False,
                      figsize = (10,8),
                      dpi = 600,
                      cbar_title = None,
                      directory = "."):
    """
    Creates a UMAP for the search space, highlighting the picked samples.
    ----------
    filename : str or Path
        Path to the CSV file containing the reaction search space.
    df_pred: pd.DataFrame
        dataframe with the predictions from either 
        ScopeBO.expected_improvement() or ScopeBO.predict_performance()
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
    pred_for_cut: Boolean
        Default = False
        Show predictions also for cut samples if True
    figsize : tuple, default=(10, 8)
        Size of the generated UMAP figure in inches.
    dpi : int, default=600
        Resolution of the output figure.
    cbar_title : str, optional
        Custom title for the colorbar. If None, uses the objective name.
    directory : str or Path, default="."
        Directory containing the CSV file.
    """

    # generate a 2D and also read in the scope information
    df_umap, df_scope, obj_name = _UMAP_creation(filename = filename,
                             obj_to_show = obj_to_show,
                             objectives = objectives,
                             directory = directory)

    # separate the observed, cut, and predicted samples
    df_seen = df_umap.loc[df_umap.index.isin(df_scope[df_scope[obj_name] != "PENDING"].index.to_list())].copy()
    df_seen[obj_name] = df_seen[obj_name].astype(float)
    prio_val = -1  # priority value corresponding to cut values
    if pred_for_cut:
        # unrealistic priority value in case predictions for cut samples are to be shown
        # will result in empty df for cut samples
        prio_val = 42  
    df_cut = df_umap[df_umap.index.isin(df_scope[df_scope["priority"] == prio_val].index.to_list())].copy()
    df_unseen = df_umap.loc[(~df_umap.index.isin(df_seen.index.to_list())) &
                            (~df_umap.index.isin(df_cut.index.to_list()))].copy()

    # determine the source of the predictions (predict_performance() or expected_improvement())
    # (using characteristic naming of the columns)
    pred_type = None
    if f"{obj_name}_pred" in df_pred.columns.values:
        pred_type = "mlr"  # from predict_performance()
    elif f"Prediction_{obj_name}" in df_pred.columns.values:
        pred_type = "ei"  # from expected_improvement()
    # assign the predicted values
    for idx in df_unseen.index:
        if pred_type == "mlr":
            df_unseen.loc[idx,obj_name] = df_pred.loc[idx,f"{obj_name}_pred"]
        elif pred_type == "ei":
            df_unseen.loc[idx,obj_name] = df_pred.loc[idx,f"Prediction_{obj_name}"]

    # plot the figure
    plt.figure(figsize=figsize, dpi = dpi, constrained_layout = True)

    colormap = cont_cmap
    if obj_bounds is None:
        vmin = df_seen[obj_name].min()
        vmax = df_seen[obj_name].max()
    else:
        vmin = obj_bounds[1]
        vmax = obj_bounds[0]
    norm = plt.Normalize(vmin,vmax)

    # set the size for predicted values based on their standard deviation if available
    # smaller points have a higher standad deviation (i. e. more uncertain predictions)
    size_vals = [40] * len(df_unseen)
    if pred_type == "ei":
        size_min, size_max = 50, 200
        std_vals = [df_pred.loc[idx,f"Std. dev. of pred._{obj_name}"] for idx in df_unseen.index]
        std_min = np.min(std_vals)
        std_max = np.max(std_vals)
        if std_max - std_min == 0:
            # no standard deviation range, set a default size
            size_vals = [(size_min+size_max)/2] * len(df_unseen)
        else:
            size_vals = [size_min + (std_max - std_val) * (size_max - size_min) / (std_max - std_min)
                    for std_val in std_vals]
    
    if not pred_for_cut:
        # plot the cut points
        plt.scatter(df_cut["UMAP1"], df_cut["UMAP2"], s=100, edgecolor="k", marker = "X",
                    color=doyle_colors[4], alpha=0.6, linewidth = 0.3, zorder=0, label = "cut")

    # plot the predicted points
    plt.scatter(df_unseen["UMAP1"], df_unseen["UMAP2"], c=df_unseen[obj_name],
                cmap=cont_cmap, norm=norm, s= size_vals, alpha=1, edgecolor='k',
                linewidth=1, marker = "o", label = "predicted",zorder = 1)
        
    # plot the observed samples
    scatter_numeric = plt.scatter(df_seen["UMAP1"], df_seen["UMAP2"], c=df_seen[obj_name],
                                    cmap=cont_cmap, norm=norm,s=250, alpha=1, edgecolor='k', 
                                    linewidth=2, label = "measured", marker = "s", zorder = 2)

    cbar = plt.colorbar(scatter_numeric)
    if cbar_title is None:
        cbar_label = obj_name.capitalize()
        cbar.set_label(cbar_label)
    else:
        cbar.set_label(cbar_title)

    plt.legend()
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.show()
    

def _UMAP_creation(filename,
              obj_to_show = None,
              objectives = None,
              directory = "."):
    
    """
    Creates a 2D UMAP for plotting.
    Returns a df with the compound names as index and the coordinates ("UMAP1", "UMAP2") as columns.
    Returns the df of the input file ("filename") after sorting the index.
    Returns the name of the objective that will be plotted.
    
    See function UMAP_view() for input variable information (above).
    """
    
    # Set directory.
    wdir = Path(directory)

    # read the results file
    df_scope = pd.read_csv(wdir / filename, index_col=0, header=0)
    df_scope.sort_index(inplace=True)
    if "priority" in df_scope.columns.to_list():
        df_scope["priority"] = df_scope["priority"].astype(float)

    # identify the objectives (containing PENDING entries) if none are given
    if objectives is None:
        objectives = df_scope.columns[df_scope.eq("PENDING").any()].to_list()
        if (objectives == []) and (obj_to_show is not None):
            print("hello")
            objectives = [obj_to_show]

    # show the first objective in the UMAP if none has been specified in the function input
    if obj_to_show is None:
        obj_to_show = objectives[0]
    
    # scale the featurization data
    df_scaled = df_scope.copy(deep=True)
    if "priority" in df_scope.columns.to_list():
        df_scaled.drop(columns=objectives + ["priority"], inplace=True)
    else:
        df_scaled.drop(columns=objectives, inplace=True)
    df_scaled = pd.DataFrame(scale(df_scaled),
                             df_scaled.index,
                             df_scaled.columns)

    # create a UMAP
    fit = UMAP(n_neighbors=40, 
               min_dist=0.7,
               n_components=2,
               metric="euclidean",
               random_state=42)
    df_umap = pd.DataFrame(fit.fit_transform(df_scaled), 
                           index = df_scaled.index,
                           columns = ["UMAP1","UMAP2"])
    df_umap.index = df_umap.index.astype(str)

    # add the result labels
    df_umap[obj_to_show] = df_scope[obj_to_show]

    return df_umap, df_scope, obj_to_show
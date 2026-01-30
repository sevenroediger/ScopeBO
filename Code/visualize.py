from pathlib import Path

from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
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


def UMAP_view(filename,
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

    # Set directory.
    wdir = Path(directory)

    # read the results file
    df_scope = pd.read_csv(wdir / filename, index_col=0, header=0)
    df_scope.sort_index(inplace=True)
    df_scope["priority"] = df_scope["priority"].astype(float)

    # identify the objectives (containing PENDING entries) if none are given
    if objectives is None:
        objectives = df_scope.columns[df_scope.eq("PENDING").any()].to_list()

    # show the first objective in the UMAP if none has been specified in the function input
    if obj_to_show is None:
        obj_to_show = objectives[0]
    
    # scale the featurization data
    df_scaled = df_scope.copy(deep=True)
    df_scaled.drop(columns=objectives + ["priority"], inplace=True)
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
    
    # add the result labels and the sample status ("priority")
    df_umap["labels"] = df_scope[obj_to_show]
    df_umap["priority"] = df_scope["priority"]

    # separate the labelled and unlabelled samples
    df_seen = df_umap.loc[df_umap["labels"] != "PENDING"].copy()
    df_seen["labels"] = df_seen["labels"].astype(float)
    df_unseen = df_umap.loc[df_umap["labels"] == "PENDING"]

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
        scope_labels = df_seen["labels"].to_list()

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
            vmin = df_seen["labels"].min()
            vmax = df_seen["labels"].max()
        else:
            vmin = obj_bounds[1]
            vmax = obj_bounds[0]
        norm = plt.Normalize(vmin,vmax)

        # plot the neutral points
        plt.scatter(
            df_neutral["UMAP1"], df_neutral["UMAP2"], s=40, 
            linewidth=0.3, edgecolor="k", color=doyle_colors[6],
            marker="o", alpha = 0.8, zorder=1)

        # check if cut samples should be highlighted
        if display_cut_samples:
            # plot the cut samples
            plt.scatter(
                df_cut["UMAP1"], df_cut["UMAP2"], s=100, edgecolor="k", marker = "X",
                color=doyle_colors[4], alpha=0.6, linewidth = 0.3, zorder=2)
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
                color=doyle_colors[2], alpha=0.8, linewidth = 1.2, zorder=4)
        else:
            # plot the cut samples the same way as the neutral samples
            plt.scatter(
                df_alt["UMAP1"], df_alt["UMAP2"], s=40, edgecolor="k", marker = "o",
                color=doyle_colors[6], alpha=0.8, linewidth = 0.3, zorder=1)
            
        # check if alternative sugegstions should be highlighted
        if display_alternatives:
            # plot the alternative samples
            plt.scatter(
                df_alt["UMAP1"], df_alt["UMAP2"], s=100*df_alt["priority"], edgecolor="k", marker = "D",
                color=doyle_colors[2], alpha=0.8, linewidth = 1.2, zorder=3)
        else:
            # plot the cut samples the same way as the neutral samples
            plt.scatter(
                df_alt["UMAP1"], df_alt["UMAP2"], s=40, edgecolor="k", marker = "o",
                color=doyle_colors[6], alpha=0.8, linewidth = 0.3, zorder=1)
        
        # plot the selected samples
        scatter_numeric = plt.scatter(df_seen["UMAP1"], df_seen["UMAP2"], c=df_seen["labels"],
                                        cmap=cont_cmap, norm=norm,s=250, alpha=1, edgecolor='k', 
                                        linewidth=2, zorder = 5)

        cbar = plt.colorbar(scatter_numeric)
        if cbar_title is None:
            cbar_label = obj_to_show.capitalize()
            cbar.set_label(cbar_label)
        else:
            cbar.set_label(cbar_title)

        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.show()

    if return_dfs:
        return {"seen": df_seen, "neutral": df_neutral, "cut": df_cut}
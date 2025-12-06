import os
import re
import shutil
from tqdm import tqdm

from IPython.display import display
import numpy as np
import pandas as pd
from morfeus.buried_volume import BuriedVolume
from morfeus.conformer import ConformerEnsemble
from morfeus.dispersion import Dispersion
from morfeus.sasa import SASA
from morfeus.xtb import XTB
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw


def calculate_morfeus_descriptors (smiles_list,
                                   filename,
                                   common_core=None,
                                   chunk_size=10,
                                   find_restart = True,
                                   starting_smiles_nr=1,
                                   chunk_label=1):
    """
    Calculate morfeus descriptors at the GFN2-xTB level for  a list of given smiles strings.
    ---------------
    smiles_list: list
        list of smiles strings
    common_core: str or None
        smarts for the common core of interest
        Default is None --> will look for the largest common substructure in the molecule
    filename: str
        path for the generated dataset
    common_core: str or None
        SMARTS for a substructure for which atom descriptors will be extracted
        If common_core=None (Default), the atom descriptors will be calculated for the
        maximum common substructure.
    chunk_size: int
        number of compounds that will be calculated in one chunk before saving the obtained data
        at the end of the run, all chunks will be concatenated.
        Default: 25
    find_restart: Boolean
        If True, the  algorithm will parse if some chunks were already calculated and auto-restart 
        with the next chunk, overwriting the starting_smiles_nr and chunk_label variables.
        NOTE: Assumes that the chunk_size was not changed
    starting_smiles_nr: int (one-indexed)
        first entry of the smiles list to be calculated
        Default: 1
        (useful for restarting in case the calculation crashes)
        NOTE: overwriting if find_restart = True
    chunk_label: int
        label for the next chunk to be calculated
        Default: 1
        (useful for restarting in case the calculation crashes)
        NOTE: overwritten if find_restart = True
    ----------------------
    Save the descriptor data under the indicated filename
    Also returns a list of pd.DataFrames for the data of the individual chunks
    as well as a pd.DataFrame for the combined Data of all calculated structures

    """

    print("This might take several minutes or even hours. Please stand by.")
    # Find common substructure and align the template
    # Create a folder to save the separate chunks of calculation results
    if not os.path.exists("./morfeus_chunks"):
        # Create the folder
        os.makedirs("./morfeus_chunks")
    template = None
    if common_core is None:
        template = _get_mcs_template_with_consistent_atom_order(smiles_list)
    else:
        template = _map_common_core_with_consistent_atom_order(smiles_list,common_core)
    print("Atom properties will be calculated for the following common substructure:")
    depiction = Draw.MolToImage(template)
    display(depiction)
    pt = Chem.GetPeriodicTable()

    results = []
    smiles_list_chunk = []
    properties = None

    # check if some chunks have already been calculated and automatically restart
    if find_restart:

        # check if ./morfeus_chunks (results folder) is empty
        last_chunk = None
        chunk_folder = "./morfeus_chunks"
        if not os.listdir(chunk_folder):
            last_chunk = 0  # no chunks run yet
        else:
            # check which chunk was run last
            pattern = re.compile(r"morfeus_chunk_(\d+)\.csv$")
            for fname in os.listdir(chunk_folder):
                match = pattern.match(fname)
                if match:
                    obs_chunk = int(match.group(1))
                    if last_chunk is None or obs_chunk > last_chunk:
                        last_chunk = obs_chunk

        # update the next chunk label and the starting_smiles_nr
        chunk_label = last_chunk + 1
        starting_smiles_nr = last_chunk * chunk_size +1

    # go through the requested smiles
    for smiles_index, smiles in enumerate(tqdm(smiles_list[(starting_smiles_nr-1):])):
        smiles_index = smiles_index + starting_smiles_nr -1
        current_results = {}
        # Generate conformer ensemble
        ce = ConformerEnsemble.from_rdkit(smiles,optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()
        if len(ce) > 5:
            ce = ce[:5]  # prune to top 5 conformers 
        # Optimize conformers
        model = {"method": "GFN2-xTB"}
        ce.optimize_qc_engine(program="xtb",model=model,procedure="geometric")
        ce.sp_qc_engine(program="xtb",model=model)
        ce.prune_energy()
        # Get the matching substructure (excluding hydrogens)
        match = ce.mol.GetSubstructMatch(template)
        substruct_atoms = [pt.GetElementSymbol(int(ce.elements[nr])) for nr in match]
        substruct_labels = _append_occurrence_numbers(substruct_atoms)

        for conformer in ce:
            props = conformer.properties
            sasa = SASA(ce.elements, conformer.coordinates)
            disp = Dispersion(ce.elements, conformer.coordinates)
            xtb = XTB(ce.elements, conformer.coordinates)

            props["SASA"] = sasa.area
            props["Volume"] = disp.volume
            props["HOMO"] = xtb.get_homo()
            props["LUMO"] = xtb.get_lumo()
            props["IP"] = xtb.get_ip(corrected=True)
            props["EA"] = xtb.get_ea(corrected=True)
            props["Dipole"] = np.linalg.norm(xtb.get_dipole())

            sasa_atom_areas = sasa.atom_areas
            disp_atom_p_int = disp.atom_p_int
            charges = xtb.get_charges()
            electrophilicity = xtb.get_fukui("electrophilicity")
            nucleophilicity = xtb.get_fukui("nucleophilicity")
            radical_fukui = xtb.get_fukui("radical")

            for i,idx in enumerate(match):
                atom_label = substruct_labels[i]
                bv = BuriedVolume(ce.elements, conformer.coordinates, idx+1)
                props[f"{atom_label}_BV"] = bv.fraction_buried_volume
                props[f"{atom_label}_SASA"] = sasa_atom_areas[idx+1]
                props[f"{atom_label}_P_int"] = disp_atom_p_int[idx+1]
                props[f"{atom_label}_charge"] = charges[idx+1]
                props[f"{atom_label}_electrophilicity"] = electrophilicity[idx+1]
                props[f"{atom_label}_nucleophilicity"] = nucleophilicity[idx+1]
                props[f"{atom_label}_radicalFukui"] = radical_fukui[idx+1]
            if smiles == smiles_list[(starting_smiles_nr-1)] and conformer == ce[0]:
                properties = props.keys()

        for property in properties:
            current_results[property] = ce.boltzmann_statistic(property)
        results.append(current_results)

        smiles_list_chunk.append(smiles)  # add the smiles to the list of smiles in this chunk

        if (smiles_index + 1) % chunk_size == 0:  # check if the chunk is full; +1 due to zero-indexing
            pd.DataFrame(results,index=smiles_list_chunk,columns=properties).to_csv(f"./morfeus_chunks/morfeus_chunk_{chunk_label}.csv",index=True,header=True)
            # clean out the collection variables for the next chunk
            results = []
            smiles_list_chunk = []
            chunk_label += 1  # update the chunk label for the next chunk
    
    # once all smiles have been calculated, save the last samples in a final chunk
    if smiles_list_chunk:
        pd.DataFrame(results,index=smiles_list_chunk,columns=properties).to_csv(f"./morfeus_chunks/morfeus_chunk_{chunk_label}.csv",index=True,header=True)

    # combine the chunks and clean up
    dfs = []
    for chunk_file in os.listdir("./morfeus_chunks"):
        dfs.append(pd.read_csv(f"./morfeus_chunks/{chunk_file}",index_col=0,header=0))
    df_combined = pd.concat(dfs,axis=0)
    df_combined.to_csv(filename,index=True,header=True)
    shutil.rmtree("./morfeus_chunks")
    os.remove("qce_optim.xyz")

    print("Finished descriptor calculation.")

    return dfs, df_combined


def _append_occurrence_numbers(strings):
    counts = {}
    result = []
    for s in strings:
        counts[s] = counts.get(s, 0) + 1
        result.append(f"{s}_{counts[s]}")
    return result


def _generate_template(ref_mol,core_mol):

    match = ref_mol.GetSubstructMatch(core_mol)

    # Create a substructure mol (just the MCS) with atom mapping
    em = Chem.EditableMol(Chem.Mol())
    atom_map = {}
    for new_idx, old_idx in enumerate(match):
        atom = ref_mol.GetAtomWithIdx(old_idx)
        atom_map[old_idx] = new_idx
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetAtomMapNum(new_idx + 1)  # Atom map numbers start at 1
        em.AddAtom(new_atom)

    # Add bonds within the MCS
    for i, idx1 in enumerate(match):
        for j, idx2 in enumerate(match):
            if idx1 >= idx2:
                continue
            bond = ref_mol.GetBondBetweenAtoms(idx1, idx2)
            if bond:
                em.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    mcs_submol = em.GetMol()
    smarts_with_map = Chem.MolToSmarts(mcs_submol)
    template = Chem.MolFromSmarts(smarts_with_map)

    return template


def _get_mcs_template_with_consistent_atom_order(smiles_list):
    # Convert SMILES to H-added mols
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles_list]

    # Find MCS
    mcs_result = rdFMCS.FindMCS(mols)
    mcs_mol = Chem.RemoveAllHs(Chem.MolFromSmarts(mcs_result.smartsString))

    # Use first molecule as reference for the template generation
    template = _generate_template(ref_mol=mols[0],core_mol=mcs_mol)
    return template


def _map_common_core_with_consistent_atom_order(smiles_list,common_core):
    # generate mol objects for smiles_list
    mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]

    # generate mol for common_core
    core_mol = Chem.RemoveAllHs(Chem.MolFromSmarts(common_core))

    # Use first molecule as reference for the template generation
    template = _generate_template(ref_mol=mols[0],core_mol=core_mol)
    return template
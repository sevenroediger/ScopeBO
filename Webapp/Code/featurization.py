import os
import re
import shutil
import logging
import gc
import glob
import multiprocessing as mp
import traceback
import queue as queue_module
import base64
from io import BytesIO
from tqdm import tqdm

from IPython.display import display
from morfeus.buried_volume import BuriedVolume
from morfeus.conformer import ConformerEnsemble
from morfeus.dispersion import Dispersion
from morfeus.sasa import SASA
from morfeus.xtb import XTB
import numpy as np
import pandas as pd
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
        Default: 10
    find_restart: Boolean
        If True, the  algorithm will parse if some chunks were already calculated and auto-restart 
        with the next chunk, overwriting the starting_smiles_nr and chunk_label variables.
        NOTE: Assumes that the chunk_size was not changed
    starting_smiles_nr: int (one-indexed)
        first entry of the smiles list to be calculated
        Default: 1
        (useful for restarting in case the calculation crashes)
        NOTE: overwritten if find_restart = True
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
    if not os.path.exists("./featurization_temp"):
        # Create the folder
        os.makedirs("./featurization_temp")

    # Create a template for the common core with consistent atom ordering
    template = None
    if common_core is None:
        # find the maximum common substructure if no common core is provided
        template = _get_mcs_template_with_consistent_atom_order(smiles_list)
    else:
        # use the provided common core
        template = _map_common_core_with_consistent_atom_order(smiles_list,common_core)

    # draw the template
    print("Atom properties will be calculated for the following common substructure:")
    depiction = Draw.MolToImage(template)
    display(depiction)

    pt = Chem.GetPeriodicTable()

    results = []
    smiles_list_chunk = []
    properties = None

    # check if some chunks have already been calculated and automatically restart
    if find_restart:

        # check if ./featurization_temp (temporary results folder) is empty
        last_chunk = None
        chunk_folder = "./featurization_temp"
        if not os.listdir(chunk_folder):
            last_chunk = 0  # no chunks run yet
        else: # some chunks already exist
            # check which chunk was run last
            pattern = re.compile(r"feat_chunk_(\d+)\.csv$")
            for fname in os.listdir(chunk_folder):
                match = pattern.match(fname)
                if match:
                    obs_chunk = int(match.group(1))
                    if last_chunk is None or obs_chunk > last_chunk:
                        last_chunk = obs_chunk

        # update the next chunk label and the starting_smiles_nr
        chunk_label = last_chunk + 1
        starting_smiles_nr = last_chunk * chunk_size +1

    # go through the provided smiles
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

        # calculate properties for each conformer
        for conformer in ce:
            props = conformer.properties
            sasa = SASA(ce.elements, conformer.coordinates)
            disp = Dispersion(ce.elements, conformer.coordinates)
            xtb = XTB(ce.elements, conformer.coordinates)

            # save the global property values
            props["SASA"] = sasa.area
            props["Volume"] = disp.volume
            props["HOMO"] = xtb.get_homo()
            props["LUMO"] = xtb.get_lumo()
            props["IP"] = xtb.get_ip(corrected=True)
            props["EA"] = xtb.get_ea(corrected=True)
            props["Dipole"] = np.linalg.norm(xtb.get_dipole())

            # calculate atom properties
            sasa_atom_areas = sasa.atom_areas
            disp_atom_p_int = disp.atom_p_int
            charges = xtb.get_charges()
            electrophilicity = xtb.get_fukui("electrophilicity")
            nucleophilicity = xtb.get_fukui("nucleophilicity")
            radical_fukui = xtb.get_fukui("radical")

            # save the atom properties
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

            # collect the property names during the first iteration
            if smiles == smiles_list[(starting_smiles_nr-1)] and conformer == ce[0]:
                properties = props.keys()

        # calculate Boltzmann-weighted average for each property
        for property in properties:
            current_results[property] = ce.boltzmann_statistic(property)
        results.append(current_results)

        smiles_list_chunk.append(smiles)  # add the smiles to the list of smiles in this chunk

        if (smiles_index + 1) % chunk_size == 0:  # check if the chunk is full; +1 due to zero-indexing
            # save the current chunk
            pd.DataFrame(results,index=smiles_list_chunk,columns=properties).to_csv(f"./featurization_temp/feat_chunk_{chunk_label}.csv",
                                                                                    index=True, header=True)
            
            # clean out the collection variables for the next chunk
            results = []
            smiles_list_chunk = []
            chunk_label += 1  # update the chunk label for the next chunk
    
    # once all smiles have been calculated, save the last samples in a final chunk
    if smiles_list_chunk:
        pd.DataFrame(results,index=smiles_list_chunk,columns=properties).to_csv(f"./featurization_temp/feat_chunk_{chunk_label}.csv",index=True,header=True)

    # combine the chunks and clean up
    dfs = []
    for chunk_file in os.listdir("./featurization_temp"):
        dfs.append(pd.read_csv(f"./featurization_temp/{chunk_file}",index_col=0,header=0))
    df_combined = pd.concat(dfs,axis=0)
    df_combined.to_csv(filename,index=True,header=True)
    shutil.rmtree("./featurization_temp")
    os.remove("qce_optim.xyz")

    print("Finished descriptor calculation.")

    return dfs, df_combined


def calculate_morfeus_descriptors_web(
    smiles_list,
    filename,
    common_core=None,
    chunk_size=5,
    worker_batch_size=5,
    find_restart=True,
    starting_smiles_nr=1,
    chunk_label=1,
    progress_callback=None,
    check_cancel=None,
    temp_dir="./featurization_temp_web",
):
    """
    Webapp-oriented variant of calculate_morfeus_descriptors.

    Differences to the notebook version:
    - No tqdm progress bar output
    - No IPython display output
    - Optional progress callback for UI updates
    - Optional cancellation callback for long-running jobs
    - Uses a separate temporary directory by default
    - Does not write a final combined csv file; returns the dataframe

    progress_callback signature:
        progress_callback(done_count, total_count, message)

    check_cancel signature:
        check_cancel() -> bool
    """

    if progress_callback is not None:
        progress_callback(0, len(smiles_list), "Setting up featurization calculation...")

    # Start from a clean slate so stale temporary files do not accumulate across runs.
    _cleanup_qcengine_artifacts()

    # For restart-enabled runs, keep existing chunks so calculation can resume.
    # For fresh runs, clear stale temporary data first.
    if os.path.exists(temp_dir) and not find_restart:
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    if common_core is None:
        template = _get_mcs_template_with_consistent_atom_order(smiles_list)
    else:
        template = _map_common_core_with_consistent_atom_order(smiles_list, common_core)

    template_smarts = Chem.MolToSmarts(template)

    mp_ctx = mp.get_context("spawn")
    worker_batch_size = max(1, int(worker_batch_size))

    results = []
    smiles_list_chunk = []
    properties = None

    if find_restart:
        last_chunk = 0
        if os.listdir(temp_dir):
            pattern = re.compile(r"feat_chunk_(\d+)\.csv$")
            for fname in os.listdir(temp_dir):
                match = pattern.match(fname)
                if match:
                    obs_chunk = int(match.group(1))
                    if obs_chunk > last_chunk:
                        last_chunk = obs_chunk

        chunk_label = last_chunk + 1
        starting_smiles_nr = last_chunk * chunk_size + 1

        if progress_callback is not None and last_chunk > 0:
            progress_callback(
                0,
                len(smiles_list),
                f"Resuming from existing chunks: found {last_chunk} completed chunk(s) of size {chunk_size}.",
            )

    smiles_slice = smiles_list[(starting_smiles_nr - 1):]
    total_count = len(smiles_slice)
    total_full_count = len(smiles_list)
    completed_before_restart = max(0, starting_smiles_nr - 1)

    if progress_callback is not None and completed_before_restart > 0:
        progress_callback(
            completed_before_restart,
            total_full_count,
            f"Processed {completed_before_restart}/{total_full_count}",
        )
    completed_successfully = False

    try:
        for batch_start in range(0, total_count, worker_batch_size):
            queue = None
            worker = None
            try:
                if check_cancel is not None and check_cancel():
                    raise RuntimeError("Featurization interrupted by user.")

                smiles_batch = smiles_slice[batch_start: batch_start + worker_batch_size]
                queue = mp_ctx.Queue()
                worker = mp_ctx.Process(
                    target=_compute_smiles_batch_descriptors_worker,
                    args=(smiles_batch, template_smarts, queue),
                )
                worker.start()
                done_marker_received = False
                processed_in_batch = 0

                while True:
                    if check_cancel is not None and check_cancel():
                        raise RuntimeError("Featurization interrupted by user.")

                    try:
                        payload = queue.get(timeout=0.2)
                    except queue_module.Empty:
                        if worker.is_alive():
                            continue
                        # Worker exited; if no explicit completion marker arrived, stop waiting.
                        if not done_marker_received:
                            break
                        continue

                    payload_type = payload.get("type")

                    if payload_type == "item":
                        smiles = payload["smiles"]
                        current_results = payload["results"]
                        child_properties = payload["properties"]

                        if properties is None:
                            properties = child_properties

                        for prop in properties:
                            if prop not in current_results:
                                current_results[prop] = np.nan

                        results.append(current_results)
                        smiles_list_chunk.append(smiles)

                        smiles_index = batch_start + processed_in_batch + starting_smiles_nr - 1
                        processed_in_batch += 1

                        if (smiles_index + 1) % chunk_size == 0:
                            pd.DataFrame(results, index=smiles_list_chunk, columns=properties).to_csv(
                                f"{temp_dir}/feat_chunk_{chunk_label}.csv", index=True, header=True
                            )
                            results = []
                            smiles_list_chunk = []
                            chunk_label += 1

                        if progress_callback is not None:
                            done_count = completed_before_restart + batch_start + processed_in_batch
                            progress_callback(
                                done_count,
                                total_full_count,
                                f"Processed {done_count}/{total_full_count}",
                            )

                    elif payload_type == "error":
                        raise RuntimeError(
                            f"Descriptor worker error for batch starting at index {batch_start + starting_smiles_nr}: "
                            f"{payload.get('error', 'unknown')}\n"
                            f"{payload.get('traceback', '')}"
                        )

                    elif payload_type == "done":
                        done_marker_received = True
                        break

                worker.join()

                if worker.exitcode != 0:
                    raise RuntimeError(
                        f"Descriptor worker failed for batch starting at index {batch_start + starting_smiles_nr} "
                        f"with exit code {worker.exitcode}."
                    )

                if not done_marker_received:
                    raise RuntimeError(
                        f"Descriptor worker did not report completion for batch starting at index "
                        f"{batch_start + starting_smiles_nr}."
                    )
            finally:
                if queue is not None:
                    try:
                        queue.close()
                        queue.join_thread()
                    except Exception:
                        pass

                if worker is not None and worker.is_alive():
                    worker.terminate()
                    worker.join()

                _close_qcengine_file_handlers()
                _cleanup_qcengine_artifacts()
                gc.collect()

        if smiles_list_chunk:
            pd.DataFrame(results, index=smiles_list_chunk, columns=properties).to_csv(
                f"{temp_dir}/feat_chunk_{chunk_label}.csv", index=True, header=True
            )

        dfs = []
        chunk_files = sorted(
            [f for f in os.listdir(temp_dir) if re.match(r"feat_chunk_\d+\.csv$", f)]
        )
        for chunk_file in chunk_files:
            dfs.append(pd.read_csv(f"{temp_dir}/{chunk_file}", index_col=0, header=0))

        df_combined = pd.concat(dfs, axis=0)

        if progress_callback is not None:
            progress_callback(total_full_count, total_full_count, "Featurization finished")

        # Match notebook behavior: clean temporary chunk folder only after successful combine.
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as cleanup_error:
                if progress_callback is not None:
                    progress_callback(
                        total_full_count,
                        total_full_count,
                        f"Warning: temporary cleanup failed ({cleanup_error}).",
                    )

        completed_successfully = True

        return dfs, df_combined

    finally:
        # Preserve temp_dir/chunks on interruption or failure for inspection/restart.
        # It is removed only after successful chunk combination above.
        if not completed_successfully and progress_callback is not None and os.path.exists(temp_dir):
            progress_callback(
                0,
                total_count,
                f"Interrupted or failed run: retained temporary chunks in {temp_dir}.",
            )

        if os.path.exists("qce_optim.xyz"):
            try:
                os.remove("qce_optim.xyz")
            except OSError:
                # Best-effort cleanup only.
                pass

        _cleanup_qcengine_artifacts()


def _compute_smiles_batch_descriptors_worker(smiles_batch, template_smarts, queue):
    """Compute descriptors for a small SMILES batch in one isolated process."""
    try:
        for smiles in smiles_batch:
            current_results, properties = _compute_single_smiles_descriptors(smiles, template_smarts)
            queue.put(
                {
                    "type": "item",
                    "smiles": smiles,
                    "results": current_results,
                    "properties": properties,
                }
            )
            _close_qcengine_file_handlers()
            _cleanup_qcengine_artifacts()
            gc.collect()

        queue.put({"type": "done"})
    except Exception as exc:
        queue.put({
            "type": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        _close_qcengine_file_handlers()
        _cleanup_qcengine_artifacts()
        gc.collect()


def _compute_single_smiles_descriptors(smiles, template_smarts):
    """Compute descriptors for one SMILES and return (results_dict, properties_list)."""
    _cleanup_qcengine_artifacts()

    template = Chem.MolFromSmarts(template_smarts)
    if template is None:
        raise ValueError("Invalid template SMARTS for descriptor calculation.")

    pt = Chem.GetPeriodicTable()
    current_results = {}
    properties = None

    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
    ce.prune_rmsd()
    ce.sort()
    if len(ce) > 5:
        ce = ce[:5]

    model = {"method": "GFN2-xTB"}
    ce.optimize_qc_engine(program="xtb", model=model, procedure="geometric")
    ce.sp_qc_engine(program="xtb", model=model)
    ce.prune_energy()

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

        for i, idx in enumerate(match):
            atom_label = substruct_labels[i]
            bv = BuriedVolume(ce.elements, conformer.coordinates, idx + 1)
            props[f"{atom_label}_BV"] = bv.fraction_buried_volume
            props[f"{atom_label}_SASA"] = sasa_atom_areas[idx + 1]
            props[f"{atom_label}_P_int"] = disp_atom_p_int[idx + 1]
            props[f"{atom_label}_charge"] = charges[idx + 1]
            props[f"{atom_label}_electrophilicity"] = electrophilicity[idx + 1]
            props[f"{atom_label}_nucleophilicity"] = nucleophilicity[idx + 1]
            props[f"{atom_label}_radicalFukui"] = radical_fukui[idx + 1]

        if properties is None and conformer == ce[0]:
            properties = list(props.keys())

    if properties is None:
        raise RuntimeError("No descriptors were generated for the given SMILES.")

    for prop in properties:
        current_results[prop] = ce.boltzmann_statistic(prop)

    return current_results, properties


def _append_occurrence_numbers(strings):
    counts = {}
    result = []
    for s in strings:
        counts[s] = counts.get(s, 0) + 1
        result.append(f"{s}_{counts[s]}")
    return result


def _close_qcengine_file_handlers():
    """Best-effort cleanup of leaked file log handlers from qcengine/geometric."""
    logger_names = {"", "geometric", "qcengine"}

    # Include all currently registered logger names to catch nested namespaces.
    manager = logging.getLogger().manager
    for name in manager.loggerDict.keys():
        if isinstance(name, str):
            logger_names.add(name)

    for name in logger_names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                finally:
                    logger.removeHandler(handler)


def _cleanup_qcengine_artifacts(base_dir="."):
    """Best-effort cleanup of transient qce/geometric/xtb artifacts."""
    file_patterns = [
        "qce_optim.xyz",
        "qce_*.json",
        "qce_*.log",
        "qce_*.txt",
        "xtbopt.xyz",
        "xtbopt.log",
        "xtbopt.coord",
        "xtbrestart",
        "xtb.trj",
        "g98.out",
        "gradient",
        "hessian",
        "charges",
        "wbo",
        ".xtboptok",
        "coord",
        "coord.original",
    ]
    dir_patterns = [
        "qce_*",
        "geometric*",
        "xtbtmp*",
    ]

    for pattern in file_patterns:
        for path in glob.glob(os.path.join(base_dir, pattern)):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    for pattern in dir_patterns:
        for path in glob.glob(os.path.join(base_dir, pattern)):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass


def build_common_core_preview(smiles_list, common_core=None, image_size=(320, 180)):
    """Build SMARTS + PNG data URI preview for the common-core template."""
    if common_core is None:
        template = _get_mcs_template_with_consistent_atom_order(smiles_list)
    else:
        template = _map_common_core_with_consistent_atom_order(smiles_list, common_core)

    template_smarts = Chem.MolToSmarts(template)
    image = Draw.MolToImage(template, size=image_size)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    image_src = f"data:image/png;base64,{encoded}"

    return template, template_smarts, image_src


def _generate_template(ref_mol,core_mol):
    """
    Generate a template mol with consistent atom ordering 
    based on a reference molecule and a common core.
    """

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
    """Generate a template mol with consistent atom ordering based on the MCS of a list of smiles strings."""

    # Convert SMILES to H-added mols
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles_list]

    # Find MCS
    mcs_result = rdFMCS.FindMCS(mols)
    mcs_mol = Chem.RemoveAllHs(Chem.MolFromSmarts(mcs_result.smartsString))

    # Use first molecule as reference for the template generation
    template = _generate_template(ref_mol=mols[0],core_mol=mcs_mol)
    return template


def _map_common_core_with_consistent_atom_order(smiles_list,common_core):
    """Generate a template mol with consistent atom ordering based on a provided common core and a list of smiles strings."""
    # generate mol objects for smiles_list
    mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]

    # generate mol for common_core
    core_mol = Chem.RemoveAllHs(Chem.MolFromSmarts(common_core))

    # Use first molecule as reference for the template generation
    template = _generate_template(ref_mol=mols[0],core_mol=core_mol)
    return template
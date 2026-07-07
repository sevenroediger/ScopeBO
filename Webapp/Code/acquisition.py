import random

from botorch.models import SingleTaskGP
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
import numpy as np
import torch

from .model import build_and_optimize_model


def explorative_run(surrogate_model, q, objective_weights,idx_test, test_x_torch):
    """
    Fully explorative acquisition function solely for benchmarking purpose.
    Input:
        surrogate_model: surrogate model object
            The surrogate model to be used.
        q: int
            batch size
        objective_weights: list of floats
            lists of the weights to be used for the scalarization in multi-obj. optimization
        idx_test: list
            list of test indices
        test_x_torch: tensor
            torch tensor of the test data features

    Returns the indices of the selected samples, their data, and their list position in the test data lists.
    """

    # get the variance of the test samples from the surrogate model
    variance = surrogate_model.posterior(test_x_torch).variance

    # scalarization in case of multi-objective optimization
    if type(variance[0].detach().tolist()) is list:  # for mono-objective, the type would be float
        # use the provided weights or otherwise average the predicted variance
        if objective_weights is not None:
            objective_weights = torch.tensor(objective_weights).to(**tkwargs).double()
            variance = (variance * objective_weights).sum(dim=-1)
        else:  # average if no weights are provided
            variance = variance.mean(dim=-1)

    # convert to numpy array
    variance_np = variance.detach().numpy()

    # sort the posterior variance
    sorted_variance = variance_np.tolist().copy()
    sorted_variance.sort(reverse=True)

    # only keep the top scores (= batch size)
    selected_variance = sorted_variance[:q]

    # determine the list indices that belong to these variance scores
    list_positions_selected_variance = []
    for current_variance in selected_variance:
        position = [i for i,x in enumerate(list(variance_np)) if x == current_variance]

        # add the positions of all list occurances of the selected variances to the position list
        for k in range(len(position)):
                if len(list_positions_selected_variance) < q:
                    list_positions_selected_variance.append(position[k])

    samples = []
    best_samples=[]
    # get the indices for these samples in idx_test
    for position in list_positions_selected_variance:
        best_samples.append(idx_test[position])
        samples.append(test_x_torch.detach().numpy().tolist()[position])

    return best_samples, samples, list_positions_selected_variance


def greedy_run(surrogate_model, q, objective_weights, idx_test, test_x_torch):
    """
    Fully exploitative acqusition function.
    If used for multi-objective optimization, the objectives are scalarized using provided weights or by averaging.
    Input:
        surrogate_model: surrogate model object
            The surrogate model to be used.
        q: int
            batch size
        objective_weights: list of float
            list with weights for the objectives in the scalarization (only multi-obj. opt.)
        idx_test: list
            list of test indices
        test_x_torch: tensor
            torch tensor of the test data features

    Returns the indices of the selected samples, their data, and their list position in the test data lists.
    """

    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    # get the surrogate means for the test samples
    means = surrogate_model.posterior(test_x_torch).mean

    # scalarization in case of multi-objective optimization
    if type(means[0].detach().tolist()) is list:  # for mono-objective, the type would be float
        # use the provided weights or otherwise average the predicted means
        if objective_weights is not None:
            objective_weights = torch.tensor(objective_weights).to(**tkwargs).double()
            means = (means * objective_weights).sum(dim=-1)
        else:  # average if no weights are provided
            means = means.mean(dim=-1)

    # convert to numpy array
    means_np = means.detach().numpy()

    # sort the posterior means
    sorted_means = means_np.tolist().copy()
    sorted_means.sort(reverse=True)

    # only keep the top scores (= batch size)
    selected_means = sorted_means[:q]

    # determine the list indices that belong to these mean scores
    list_positions_selected_means = []
    for current_means in selected_means:
        position = [i for i,x in enumerate(list(means_np)) if x == current_means]

        # add the positions of all list occurances of the selected means to the position list
        for k in range(len(position)):
                if len(list_positions_selected_means) < q:
                    list_positions_selected_means.append(position[k])

    samples = []
    best_samples=[]
    # get the indices for these samples in idx_test
    for position in list_positions_selected_means:
        best_samples.append(idx_test[position])
        samples.append(test_x_torch.detach().numpy().tolist()[position])

    return best_samples, samples, list_positions_selected_means


def hypervolume_improvement(surrogate_model, q, objective_weights, cumulative_train_y, idx_test, test_x_torch):
    """
    Multiobjective greedy acquisition function. Exploit the improvement of the current hypervolume.
    Input:
        surrogate_model: surrogate model object
            The surrogate model to be used.
        q: int
            batch size
        objective_weights: list of float
            list with weights for the objectives in the scalarization (only multi-obj. opt.)
        cumulative_train_y: np.array
            array of the training data labels
        idx_test: list
            list of test indices
        test_x_torch: tensor
            torch tensor of the test data features

    Returns the indices of the selected samples, their data, and their list position in the test data lists.
    """
     
    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    # Calculate the reference point for the hypervolume (minimum value for each objective that was seen so far).
    ref_mins = np.min(cumulative_train_y, axis=0)
    ref_point = torch.tensor(ref_mins).double().to(**tkwargs)

    # Get the surrogate means for the test samples
    means = surrogate_model.posterior(test_x_torch).mean
    
    # Get the current Pareto front
    train_y_torch = torch.tensor(cumulative_train_y).to(**tkwargs).double()
    pareto_mask = is_non_dominated(train_y_torch)  # Check which points are non-dominated
    pareto_y = train_y_torch[pareto_mask]  # The Pareto front is the collection of non-dominated points.

    # Apply weights to the predicted means and Pareto front if requested.
    if objective_weights is not None:
        objective_weights = torch.tensor(objective_weights).to(**tkwargs).double()
        means = means * objective_weights
        pareto_y = pareto_y * objective_weights

    # Calculate the hypervolume of the current Pareto front.
    hv = Hypervolume(ref_point=ref_point)  # initialize hypervolume object.
    initial_hv = hv.compute(pareto_y)  # calculate hypervolume
    
    # Calculate the improvement of the hypervolume for each test set point
    hv_improvements = []
    for i in range(means.shape[0]):  # loop through all test set samples
        current_front = torch.cat([pareto_y, means[i].unsqueeze(0)], dim=0)  # add the test sample to the current pareto front
        current_hv = hv.compute(current_front[is_non_dominated(current_front)])  # calculate the hv for that front
        improvement = current_hv - initial_hv  # get the hypervolume improvement
        hv_improvements.append(improvement)

    # Get the indices of the top q values in hv_improvements (q is the batch size)
    top_q_indices = np.array(hv_improvements).argsort()[-q:][::-1]

    # Remove indices for which the hv_improvement was 0 (in case not enough led to improvement)
    list_positions_samples = [idx for idx in top_q_indices if hv_improvements[idx] > 0]

    # get the samples from the test data
    test_x_list = test_x_torch.detach().numpy().tolist()
    samples = [test_x_list[position] for position in list_positions_samples]
    best_samples= [idx_test[position] for position in list_positions_samples]

    # use the greedy acq fct to calculate the remaining samples in case that
    # less samples than requested improved the hypervolume
    # this acq fct uses simple scalarization of the objectives instead of hypervolume improvement
    remaining_q = q - len(list_positions_samples)
    if remaining_q != 0:
        # remove the just selected samples from idx_test and test_x_torch to avoid duplicate selections
        updated_idx_test = np.delete(idx_test, list_positions_samples)
        for position in list_positions_samples:
            test_x_list.pop(position)
        idx_greedy, samples_greedy, list_positions_greedy = greedy_run(
            surrogate_model=surrogate_model, q=remaining_q, objective_weights=objective_weights,
            idx_test=updated_idx_test, test_x_torch=torch.tensor(test_x_list).to(**tkwargs).double())
        
        # Apped these results to the results from the hypervolume acq fct
        for idx in idx_greedy:
            best_samples.append(idx)
        for sample in samples_greedy:
            samples.append(sample)
        for position in list_positions_greedy:
            list_positions_samples.append(position)

    return best_samples, samples, list_positions_samples


def random_run(q, idx_test, seed):
    """
    Acquisition function using seeded random selection solely for benchmarking purpose.
    Input:
        q: int
            batch size
        idx_test: list
            list of test indices
        seed: int
            random seed for reproducibility
        
    Returns the indices of the selected samples and their list position in the test data lists.
    """

    # Set the random seed.
    random.seed(seed)

    # Select q datapoints.
    best_samples = random.sample(list(idx_test),q)

    # Get the corresponding list positions.
    list_positions_selected_samples = []
    for sample in best_samples:
        list_positions_selected_samples.append(list(idx_test).index(sample))

    return best_samples, list_positions_selected_samples


def low_variance_selection(batch, idx_test, cumulative_test_x, cumulative_train_x, cumulative_train_y):
    """
    Acqusition function using selection of the points with the lowest variance for benchmarking purposes (no batch fantasy). 
    Only works for single objectives.

    Returns the indices of the selected samples.

    Input:
        batch: int
            batch size
        idx_test: list
            list of test indices
        cumulative_test_x: np.array
            array of the test data features
        cumulative_train_x: np.array
            array of the training data features
        cumulative_train_y: np.array
            array of the training data labels
    """

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cpu")}

    # Tensors for input data.
    train_x_torch = torch.tensor(cumulative_train_x).to(**tkwargs).double()
    test_x_torch = torch.tensor(cumulative_test_x).double().to(**tkwargs)
    train_y_torch = torch.tensor(cumulative_train_y).double().to(**tkwargs)
    
    # Optimize GP model using function from model.py.
    gp, likelihood = build_and_optimize_model(train_x=train_x_torch, train_y=train_y_torch)

    # Creating a single task GP model for the objective and storing it in individual_models list.
    surrogate_model = SingleTaskGP(train_X=train_x_torch, train_Y=train_y_torch,
                        covar_module=gp.covar_module, likelihood=likelihood)
    
    # delete variables that are not required anymore to save memory
    del gp
    del likelihood

    # get the variance of the test samples from the surrogate model
    variance = surrogate_model.posterior(test_x_torch).variance.detach().numpy()

    # sort the posterior variance
    sorted_variance = variance.tolist().copy()
    sorted_variance.sort(reverse=False)  # ascending order

    # only keep the lowest scores (= batch size)
    selected_variance = sorted_variance[:batch]

    # determine the list indices that belong to these variance scores
    position = None
    list_positions_selected_variance = []

    for j in range(len(selected_variance)):
        position = [i for i,x in enumerate(list(variance)) if x == selected_variance[j]]

        # add the positions of all list occurances of the selected variance to the position list
        for k in range(len(position)):
                if len(list_positions_selected_variance) < batch:
                    list_positions_selected_variance.append(position[k])

    # get the indices for these samples in idx_test
    best_samples = idx_test[list_positions_selected_variance]

    return best_samples
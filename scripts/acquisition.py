import random

from botorch.models import SingleTaskGP
import torch

from .model import build_and_optimize_model


def explorative_run(surrogate_model, q, idx_test, test_x_torch):
    """
    Fully explorative acquisition function solely for benchmarking purpose. Only works for mono-objective optimization.
    Input:
        surrogate_model: surrogate model object
            The surrogate model to be used.
        q: int
            batch size
        idx_test: list
            list of test indices
        test_x_torch: tensor
            torch tensor of the test data features

    Returns the indices of the selected samples, their data, and their list position in the test data lists.
    """

    variance = surrogate_model.posterior(test_x_torch).variance.detach().numpy()

    # sort the posterior variance
    sorted_variance = variance.tolist().copy()
    sorted_variance.sort(reverse=True)

    # only keep the top scores (= batch size)
    selected_variance = sorted_variance[:q]

    # determine the list indices that belong to these variance scores
    list_positions_selected_variance = []
    for current_variance in selected_variance:
        position = [i for i,x in enumerate(list(variance)) if x == current_variance]

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


def greedy_run(surrogate_model, q, objective_mode, idx_test, test_x_torch):
    """
    Fully exploitative acqusition function.
    Input:
        surrogate_model: surrogate model object
            The surrogate model to be used.
        q: int
            batch size
        objective_weights: list of float
            list with weights for the objectives in the scalarization (only multi-obj. opt.)
        objective_mode: list of str
            list with the modes of the individual objectives (maximization or minimization)
        idx_test: list
            list of test indices
        test_x_torch: tensor
            torch tensor of the test data features

    Returns the indices of the selected samples, their data, and their list position in the test data lists.
    """

    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    # get the surrogate means
    means = surrogate_model.posterior(test_x_torch).mean

    # check for minimization tasks and adjust values if so
    # sign change converts minimization problems to pseudo-maximization problem for the algorithm
    obj_sign = [-1.0 if mode.lower() == "min" else 1.0 for mode in objective_mode]
    means = means * torch.tensor(obj_sign).to(**tkwargs).double()

    # scalarization in case of multi-objective optimization
    if len(obj_sign) > 1:
        # use the provided weights or otherwise average the predicted means
        if objective_weights is not None:
            objective_weights = torch.tensor(objective_weights).to(**tkwargs).double()
            means = (means * objective_weights).sum(dim=-1)
        else:
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


def random_run(q, idx_test, seed):
    """
    Acquisition function using seeded random selection solely for benchmarking purpose.
    Input:
        q: int
            batch size
        idx_test: list
            list of test indices
        
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
    Acqusition function using selection of the points with the lowest variance for benchmarking purposes (no batch fantasy). Only works for single objectives.
    NOTE: Complete docstring.
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
    
    # delete variables that are not required anymore
    del gp
    del likelihood

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
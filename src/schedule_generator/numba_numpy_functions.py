import numpy as np
import numba as nb


@nb.njit
def nb_set_seed(seed: int):
    np.random.seed(seed)


def select_random_item(items, probabilities: np.ndarray | None = None):
    if probabilities is None:
        return items[np.random.randint(len(items))]
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)
    if not isinstance(items, np.ndarray):
        items = np.array(items)
    return nb_select_random_item(items, probabilities)


#@nb.njit(
#    "int32(int32[:], float64[:])"
#)  # input signature is optional (helps speed up the very first call)
def nb_select_random_item(items, probabilities: np.ndarray):
    if probabilities is None:
        return items[np.random.randint(len(items))]
    # Ensure probabilities sum to 1
    prob_sum = np.sum(probabilities)
    if prob_sum != 1.0:
        probabilities = probabilities / prob_sum

    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(probabilities)

    # Generate a random number and find the corresponding item
    rand = np.random.random()
    for i in range(len(cumulative_probs)):
        if rand < cumulative_probs[i]:
            return items[i]

    # Fallback in case of rounding errors, should not happen
    return items[-1]

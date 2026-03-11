import numpy as np
from numba import njit
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair


@njit
def compute_delta(A: np.ndarray, B: np.ndarray, p: np.ndarray, i: int, j: int):
    """Computer the cost difference if elements i and j are transposed
    in permutation (solution) p.

    Based on [1] and [2]. The two instances we're doing (kra30a and
    tai40a) are symmetric, so we use the simpler calculation.

    [1]: https://cs.uwaterloo.ca/~dtompkin/archive/ubcsat/algorithms/Taillard91.pdf
    [2]: https://web.archive.org/web/20190208122324/http://mistic.heig-vd.ch/taillard/codes.dir/tabou_qap2.c
    """
    # Not identical to Equation 1 of [1]. We want negative value for
    # improvement, but original formulation would give us positive.
    # This iwhy we do B[p[j], p[k]] on left instead of B[p[i], p[k]].
    # N.b., this is also what Taillard's code in [2] does.
    n = len(p)
    d = 0
    pi, pj = p[i], p[j]
    for k in range(n):
        if k != i and k != j:
            pk = p[k]
            d += (A[i, k] - A[j, k]) * (B[pj, pk] - B[pi, pk])
    return 2 * d


@njit
def tabu_search(
    p_in: np.ndarray, A: np.ndarray, B: np.ndarray, max_iter: int, seed: int
):
    """Pass in a `seed` generated from the problem's `random_state`.
    Numba lets you pass a np.random.Generator, but it can be messy, so
    prob easier to pass a seed.
    """
    np.random.seed(seed)
    n = len(p_in)
    p = p_in.copy()
    # 8*n is passed as duration in Taillard's main()
    tabu_duration = 8 * n

    D = np.zeros((n, n))
    for i in range(n):
        # Taillard does it for range(0, i), doesn't matter b/c symmetry.
        for j in range(i + 1, n):
            D[i, j] = compute_delta(A, B, p, i, j)
            D[j, i] = D[i, j]

    tabu_list = np.zeros((n, n))
    current_cost = 0
    for i in range(n):
        for j in range(n):
            tabu_list[i, j] = -(n * i + j)
            current_cost += A[i, j] * B[p[i], p[j]]

    best_p = p.copy()
    best_cost = current_cost
    for current_iter in range(1, max_iter + 1):
        r, s = -1, -1
        min_delta = np.inf
        for i in range(n - 1):
            for j in range(i + 1, n):
                authorized = (tabu_list[i, p[j]] < current_iter) or (
                    tabu_list[j, p[i]] < current_iter
                )
                # Taillard does aspiration checks by time and priority that we
                # don't need to care about, let's just get the best one
                if authorized or (current_cost + D[i, j] < best_cost):
                    if D[i, j] < min_delta:
                        min_delta = D[i, j]
                        r, s = i, j

        # If everything is taboo, just break
        if r == -1:
            break

        p[r], p[s] = p[s], p[r]
        pr, ps = p[r], p[s]  # Less referencing in the for loop below
        current_cost += D[r, s]
        tabu_list[r, ps] = current_iter + int((np.random.random() ** 3) * tabu_duration)
        tabu_list[s, pr] = current_iter + int((np.random.random() ** 3) * tabu_duration)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if i != r and i != s and j != r and j != s:
                    # Symmetric so we can simplify with 2*
                    delta_update = (
                        2
                        * (A[r, i] - A[r, j] + A[s, j] - A[s, i])
                        * (B[ps, p[i]] - B[ps, p[j]] + B[pr, p[j]] - B[pr, p[i]])
                    )
                    D[i, j] += delta_update
                    D[j, i] = D[i, j]
                else:
                    D[i, j] = compute_delta(A, B, p, i, j)
                    D[j, i] = D[i, j]
    return best_p


class TabuSearchRepair(Repair):
    def __init__(self, max_iter=100):
        super().__init__()
        self.max_iter = max_iter

    def _do(self, problem, X, random_state: np.random.Generator = None, **kwargs):
        for i in range(len(X)):
            seed = random_state.integers(0, high=1000000)
            X[i] = tabu_search(X[i], problem.A, problem.B, self.max_iter, seed)
        return X


class SwapMutation(Mutation):
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, random_state: np.random.Generator = None, **kwargs):
        # This doens't seem documented, but think `random_state` is passed by
        # Mutation.do(), and it uses the seed we put in minimize()
        Y = X.copy()
        for i, y in enumerate(X):
            if random_state.random() < self.prob:
                a, b = random_state.choice(len(y), size=2, replace=False)
                Y[i, a], Y[i, b] = Y[i, b], Y[i, a]
        return Y


# TODO: try mixing swap and inversion, or add shuffle?

import numpy as np
from pymoo.core.mutation import Mutation


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

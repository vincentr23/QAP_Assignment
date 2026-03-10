import numpy as np
from pymoo.core.problem import Problem


class QAP(Problem):
    def __init__(self, n: int, A: np.ndarray, B: np.ndarray):
        """Defines metadata for a QAP instance.

        Args:
            n: Number of facilities or locations in the problem
            A: Flow matrix, where a_{ik} is the flow from facility i to
                facility k
            B: Distnace matrix, where b_{jl} is the distnace from
                location j to location l
        """
        super().__init__(n_var=n, n_obj=1, xl=0, xu=n - 1, vtype=int)
        self.A = A
        self.B = B

    def _evaluate(self, x, out, *args, **kwargs):
        # NumPy docs say setting `count` is better
        costs = np.fromiter(
            (np.sum(self.A * self.B[p][:, p]) for p in x),
            dtype=np.int64,
            count=len(x),
        )
        # out['F'] neeeds to be (len(x), self.n_obj)
        out["F"] = costs.reshape(-1, 1)

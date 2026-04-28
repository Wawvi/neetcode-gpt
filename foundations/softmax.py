import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        z_max = max(z)
        z = z - z_max
        z_softmax = np.exp(z)
        z_softmax = z_softmax / np.sum(z_softmax)

        return np.round(z_softmax, 4)
        

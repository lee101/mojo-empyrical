import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

# empyrical 0.5.5 predates NumPy 2.0, which renamed this constant.
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

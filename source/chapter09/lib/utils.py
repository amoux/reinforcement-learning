from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


@dataclass
class Batch:
    state: List[np.ndarray] = field(default_factory=list, repr=False)
    action: List[int] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    q_vals: List[Any] = field(default_factory=list)

# This code is part of qencore.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from .nn import MLP, PlainICNN
from .quantum_circuits import VQLS
from .splines import MultisegmentConvexInterpolation1D

__name__ = "qencore"
__version__ = "0.1.0"
__all__ = ["MLP", "PlainICNN", "VQLS", "MultisegmentConvexInterpolation1D"]

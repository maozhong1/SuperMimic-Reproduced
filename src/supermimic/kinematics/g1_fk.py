"""Backward-compatible alias (the FK is robot-generic, see ``fk.py``)."""

from supermimic.kinematics.fk import HumanoidKinematics as G1Kinematics  # noqa: F401
from supermimic.kinematics.fk import HumanoidKinematics  # noqa: F401

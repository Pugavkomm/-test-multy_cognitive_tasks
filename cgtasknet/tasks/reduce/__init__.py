"""
In reduced problems, we use two modes, which
are quoted in two different directions. Some
of the tasks can only be transferred to one mod.
The contextual task is transferred to two modes at once.
The network must ignore the wrong mod.
"""
from __future__ import annotations

from .ctxdm import *
from .dm import *
from .multy import *
from .romo import *
from .reduce_task import ReduceTaskCognitive, ReduceTaskParameters

__all__ = [
    "ReduceTaskParameters",
    "ReduceTaskCognitive",
    "DMTaskParameters",
    "RomoTaskParameters",
    "CtxDMTaskParameters",
    "DMTaskRandomModParameters",
    "RomoTaskRandomModParameters",
    "ReduceTaskCognitive",
    "DMTask",
    "DMTaskRandomMod",
    "DMTask1",
    "DMTask2",
    "RomoTask",
    "RomoTaskRandomMod",
    "RomoTask1",
    "RomoTask2",
    "CtxDMTask",
    "CtxDM1",
    "CtxDM2",
    "MultyReduceTasks",
]

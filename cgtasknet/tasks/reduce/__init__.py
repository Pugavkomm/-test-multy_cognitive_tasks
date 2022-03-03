"""
In reduced tasks, we use two modes, which
are quoted in two different directions. Some
of the tasks can only be transferred to one mod.
The contextual task is transferred to two modes at once.
The network must ignore the wrong mod.
"""
from __future__ import annotations

from .ctxdm import *
from .dm import *
from .go import (
    GoDlTask,
    GoDlTask1,
    GoDlTask2,
    GoDlTaskParameters,
    GoDlTaskRandomMod,
    GoDlTaskRandomModParameters,
    GoRtTask,
    GoRtTask1,
    GoRtTask2,
    GoRtTaskParameters,
    GoRtTaskRandomMod,
    GoRtTaskRandomModParameters,
    GoTask,
    GoTask1,
    GoTask2,
    GoTaskParameters,
    GoTaskRandomMod,
    GoTaskRandomModParameters,
)
from .multy import *
from .reduce_task import ReduceTaskCognitive, ReduceTaskParameters, _generate_values
from .romo import *

__all__ = [
    "GoTask",
    "GoTask1",
    "GoTask2",
    "GoTaskParameters",
    "GoTaskRandomMod",
    "GoTaskRandomModParameters",
    "GoRtTask",
    "GoRtTask1",
    "GoRtTask2",
    "GoRtTaskParameters",
    "GoRtTaskRandomMod",
    "GoRtTaskRandomModParameters",
    "GoDlTask",
    "GoDlTask1",
    "GoDlTask2",
    "GoDlTaskParameters",
    "GoDlTaskRandomMod",
    "GoDlTaskRandomModParameters",
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
    "_generate_values",
]

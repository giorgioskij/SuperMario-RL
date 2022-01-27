from enum import IntEnum, auto

class Action(IntEnum):
    NOOP        = 0
    RIGHT       = auto()
    RIGHT_A     = auto()
    RIGHT_B     = auto()
    RIGHT_A_B   = auto()
    A           = auto()
    LEFT        = auto()
    LEFT_A      = auto()
    LEFT_B      = auto()
    LEFT_A_B    = auto()
    DOWN        = auto()
    UP          = auto()
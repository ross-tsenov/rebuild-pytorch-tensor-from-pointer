from enum import Enum, auto

import torch


class RunningType(Enum):
    DATACLASS_METADATA = auto()
    JSON_METADATA = auto()
    FULL_TENSOR = auto()


SERVER_PROTOCOL: str = r"tcp"
SERVER_ADDRESS: str = r"0.0.0.0"
SERVER_PORT: int = 6000
TENSOR_SIZE: tuple[int, ...] = (1920, 1080, 3)
TENSOR_DTYPE: torch.dtype = torch.float
TENSOR_DEVICE: str = "cuda:0"
NUMBER_OF_ITERATION: int = 100000
RUNNING_TYPE: RunningType = RunningType.DATACLASS_METADATA

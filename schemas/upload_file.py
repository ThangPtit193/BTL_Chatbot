from enum import Enum


class StoreDocOption(str, Enum):
    default = "default"
    skip = "skip"
    overwrite = "overwrite"

from enum import Enum


class FreezeMode(Enum):
    none = 0
    base = 1
    all = 2


class DownloadMode(Enum):
    none = 0
    web = 1
    s3 = 2

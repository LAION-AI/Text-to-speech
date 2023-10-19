from os import path as osp
from environs import Env

from utils.helpers import NestedNamespace


env = Env()
env.read_env()

DIR_PATH = osp.dirname(osp.realpath(__file__))
ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), ".." + osp.sep))


settings = {
    "CONSTANTS": {},
    "ROOT_PATH": ROOT_PATH,
    "DEBUG": env.bool("DEBUG", False),
    "LOG_LEVEL": env.str("LOG_LEVEL", "DEBUG"),
    "LOG_DIR": env.str("LOG_DIR", osp.join(ROOT_PATH, "logs")),
    "DATA_DIR": env.str("DATA_DIR", osp.join(ROOT_PATH, "data")),
    "CACHE_DIR": env.str("CACHE_DIR", osp.join(ROOT_PATH, "cache")),
    "database": {
        "sqlite": {"DB": env.str("SQLITE_DB", osp.join(ROOT_PATH, "dbs", "graph.db"))}
    },
    "aws": {
        "KEY": env.str("AWS_KEY", None),
        "SECRET": env.str("AWS_SECRET", None),
        "SAVE_BUCKET": env.str("S3_SAVE_BUCKET"),
    },
    "huggingface": {"HF_TOKEN": env.str("HF_TOKEN")},
    "ray": {"RAY_HEAD_URL": env.str("RAY_HEAD_URL")},
}


settings = NestedNamespace(settings)

from hashlib import md5, sha1, sha256
import json
import random
import time
import importlib
from os.path import isfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from bson import ObjectId
from pydantic import BaseModel

EXCLUDED_SPECIAL_FIELDS = "exclude_special_fields"


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = NestedNamespace(value)
            elif isinstance(value, list):
                value = tuple(NestedNamespace(val) if isinstance(val, (dict, list)) else val for val in value)
            setattr(self, key, value)


class SpecialExclusionBaseModel(BaseModel):
    _special_exclusions: set[str]

    def dict(self, **kwargs):
        exclude = kwargs.get("exclude", {})
        if EXCLUDED_SPECIAL_FIELDS in exclude:
            exclude = {k: v for k, v in super().dict(**kwargs).items() if k not in self._special_exclusions}
        return super().dict(exclude=exclude, **kwargs)


def default(value, d):
    return value if value is not None else (d() if callable(d) else d)

def exists(value):
    return value is not None

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module), cls)


def freeze(o):
    return frozenset((k, freeze(v)) for k, v in o.items()) if isinstance(o, dict) else (tuple(freeze(v) for v in o) if isinstance(o, list) else o)


def make_hash(data, algorithm="sha1", serializer=None, sort=False):
    algos = {"sha1": sha1, "sha256": sha256, "md5": md5}
    data = default(data, "")
    data = sorted(data) if sort else data
    data = json.dumps(data).encode("utf-8") if serializer == "json" else (freeze(data) if serializer == "freeze" else data.encode("utf-8"))
    return algos[algorithm.lower()](data).hexdigest()


def generate_oid(serialize=True):
    return str(ObjectId()) if serialize else ObjectId()

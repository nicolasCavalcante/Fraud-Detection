import json
from functools import partial
from pathlib import Path
from typing import Callable, List

import mlflow
from joblib import Memory

PACKAGE_PATH = Path(__file__).parent
OUTPUT_PATH = PACKAGE_PATH.parent / 'output'
DATA_DIR = PACKAGE_PATH.parent / 'data'
MLFLOW_DIR = OUTPUT_PATH / 'mlflow'
CACHE_PATH = OUTPUT_PATH / '.cache/util.json'
memory = Memory(OUTPUT_PATH / '.cache/joblib', verbose=0)
mlflow.set_tracking_uri('file:' + MLFLOW_DIR.as_posix())


class Cache():
    def __init__(self) -> None:
        if not CACHE_PATH.is_file():
            with CACHE_PATH.open('w') as f:
                f.write('{ }')
        with CACHE_PATH.open() as f:
            data = json.load(f)
        self.data = {}
        for pathstr in data:
            self.data[Path(pathstr).resolve()] = data[pathstr]

    def changed(self, path: Path):
        path = path.resolve()
        if path in self.data:
            return path.stat().st_mtime > self.data[path]
        self.data[path.resolve()] = 1
        return True

    def update(self):
        for path in self.data:
            path: Path
            if path.exists():
                self.data[path.resolve()] = path.stat().st_mtime
        data = {}
        for path in self.data:
            data[path.as_posix()] = self.data[path]
        with CACHE_PATH.open('w') as f:
            json.dump(data, f, indent=4)


def make(deps: List[Path] = [],
         targets: List[Path] = [],
         add_funcpath_dep=True):
    def decorator(func: Callable):
        def wrapper():
            cache = Cache()
            alldeps = deps.copy()
            if add_funcpath_dep:
                if type(func) is partial:
                    fpath = Path(func.func.__code__.co_filename)
                else:
                    fpath = Path(func.__code__.co_filename)
                alldeps.append(fpath)
            can_run = all([path.exists() for path in alldeps])
            if not can_run:
                sucessfull = False
            else:
                changed = any([cache.changed(path) for path in alldeps])
                changed |= any([not path.exists() for path in targets])
                if not changed:
                    sucessfull = True
                else:
                    sucessfull = func(alldeps, targets)
            cache.update()
            return sucessfull

        return wrapper

    return decorator

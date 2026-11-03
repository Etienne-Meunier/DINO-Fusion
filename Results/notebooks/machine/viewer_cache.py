"""
Disk cache of the per-run computations of the viewer builders, in `.cache/<namespace>/` next to this file.

An entry is keyed by the files it reads (path, size, modification time), the source of the builder module above its
`if __name__ == '__main__':` block (editing a function or a constant recomputes, editing the main loop does not), the
source of the functions it imports from other modules (`source_hash`) and any extra arguments. Delete `.cache/` (or a
namespace) to force a full rebuild.
"""
import hashlib
import inspect
import json
import os
import pickle
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / '.cache'


def code_hash(*module_files):
    """Hash of the source of each of `module_files` above its `if __name__ == '__main__':` line (the whole file without one)."""
    source = ''.join(Path(f).read_text().split("if __name__ == '__main__':")[0] for f in module_files)
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def source_hash(*objects):
    """Hash of the source of functions, methods or classes, and of the repr of any other value (an imported constant)."""
    source = ''.join(inspect.getsource(o) if inspect.isroutine(o) or inspect.isclass(o) else repr(o) for o in objects)
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def array_id(array):
    """Hash of the dtype, shape and bytes of an array (numpy or torch), for inputs held in memory rather than in a file."""
    a = np.ascontiguousarray(array.cpu().numpy() if hasattr(array, 'cpu') else array)
    return hashlib.sha1(str((a.dtype, a.shape)).encode() + a.tobytes()).hexdigest()[:16]


def file_id(path):
    """(absolute path, size, mtime) of a file, the part of a cache key that changes when the file is rewritten."""
    s = os.stat(path)
    return [os.path.abspath(path), s.st_size, s.st_mtime_ns]


def cached(namespace, key, compute):
    """
    Return compute() and store it, or the stored value for the same `key` (anything json-serialisable, usually
    [code_hash(...), file_id(...), ...]). The value is pickled.
    """
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = CACHE / namespace / f'{digest}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return value

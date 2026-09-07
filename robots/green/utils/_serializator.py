"""
Configuration Serialization Utilities

This module provides utilities to serialize and deserialize Python objects
(including dataclasses, functions, classes, enums, numpy arrays, etc.)
to and from JSON format while preserving type information.

Supports:
- Dataclasses (instances)
- Functions (top-level only)
- Classes (types)
- Enums
- Numpy arrays and scalars (optional)
- pathlib.Path
- bytes/bytearray
- complex numbers
- sets, tuples
- functools.partial
- Generic objects with __dict__

Designed for configuration persistence and portability.
"""

# _serializator.py
# Python 3.10+

import base64
import dataclasses as dc
import enum
import functools
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, List

# Optional numpy support (safe if missing in clean env)
try:
    import numpy as np
except Exception:
    np = None


# ---------- Internal helpers ----------

def _tag(tag: str, payload: Any) -> Dict[str, Any]:
    """Create a tagged dictionary for type reconstruction."""
    return {tag: payload}


def _import_qualified(module: str, qualname: str) -> Any:
    """Import and return an object using its module and qualified name."""
    mod = importlib.import_module(module)
    obj = mod
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _encode_type(t: type) -> Dict[str, Any]:
    """Encode a type object for serialization."""
    return _tag("__py_type__", {"module": t.__module__, "qualname": t.__qualname__})


def _decode_type(payload: dict) -> type:
    """Decode a type object from serialized data."""
    return _import_qualified(payload["module"], payload["qualname"])


def _encode_func(fn) -> Dict[str, Any]:
    """
    Encode a function for serialization.
    
    Only top-level functions are supported (not lambdas or nested functions).
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    name = getattr(fn, "__name__", None)
    if not module or not (qualname or name):
        raise TypeError(f"Function {fn!r} is not importable")
    qn = qualname or name
    if "<locals>" in qn:
        raise TypeError(
            f"Function {fn!r} appears to be nested/lambda ({qn}). "
            "Move it to a top-level def so it can be imported."
        )
    return _tag("__py_func__", {"module": module, "qualname": qn})


def _decode_func(payload: dict) -> Any:
    """Decode a function from serialized data."""
    return _import_qualified(payload["module"], payload["qualname"])


# ---------- Encode to JSON-safe structure ----------

def _postprocess_after_deserialization(obj: Any) -> Any:
    """
    Postprocess object after from_plain.
    Currently handles: resolving tagged asset_path to absolute based on current project root.
    """
    if isinstance(obj, dict):
        if "__asset_path__" in obj:
            rel_path = obj["__asset_path__"]
            try:
                project_root = Path(__file__).parent.parent.parent.resolve()
                abs_path = project_root / "dependencies" / rel_path
                if not abs_path.exists():
                    print(f"[Asset Resolver] Asset not found: {abs_path}")
                return str(abs_path)
            except Exception as e:
                print(f"[Asset Resolver] Failed to resolve '{rel_path}': {e}")
                return str(rel_path)
        # Recurse
        return {k: _postprocess_after_deserialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_postprocess_after_deserialization(x) for x in obj]
    return obj

def to_plain(o: Any) -> Any:
    """
    Convert a Python object to a JSON-serializable structure.
    
    Preserves type information through tagged dictionaries for reconstruction.
    """
    # primitives
    if o is None or isinstance(o, (bool, int, float, str)):
        return o

    # numpy scalars
    if np is not None and isinstance(o, np.generic):
        return o.item()

    # functions/methods/builtins
    if inspect.isfunction(o) or inspect.ismethod(o) or inspect.isbuiltin(o):
        return _encode_func(o)

    # functools.partial
    if isinstance(o, functools.partial):
        return _tag(
            "__partial__",
            {
                "func": to_plain(o.func),
                "args": [to_plain(a) for a in o.args],
                "keywords": {k: to_plain(v) for k, v in (o.keywords or {}).items()},
            },
        )

    # classes (types)
    if isinstance(o, type):
        return _encode_type(o)

    # enums
    if isinstance(o, enum.Enum):
        t = type(o)
        return _tag(
            "__enum__",
            {"module": t.__module__, "qualname": t.__qualname__, "value": o.value},
        )

    # dataclasses (instances)
    if dc.is_dataclass(o) and not isinstance(o, type):
        cls = type(o)
        fields = {}
        for f in dc.fields(o):
            fields[f.name] = to_plain(getattr(o, f.name))
        return _tag(
            "__dataclass__",
            {
                "module": cls.__module__,
                "qualname": cls.__qualname__,
                "fields": fields,
            },
        )

    # numpy arrays
    if np is not None and isinstance(o, np.ndarray):
        return _tag(
            "__ndarray__",
            {"data": o.tolist(), "dtype": str(o.dtype), "shape": o.shape},
        )

    # pathlib.Path
    if isinstance(o, Path):
        return _tag("__path__", str(o))

    # bytes
    if isinstance(o, (bytes, bytearray)):
        return _tag("__bytes__", base64.b64encode(bytes(o)).decode("ascii"))

    # complex
    if isinstance(o, complex):
        return _tag("__complex__", [o.real, o.imag])

    # sets / tuples
    if isinstance(o, set):
        return _tag("__set__", [to_plain(x) for x in o])
    if isinstance(o, tuple):
        return _tag("__tuple__", [to_plain(x) for x in o])

    # dicts
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if not isinstance(k, str):
                k = repr(k)  # JSON keys must be strings
            out[k] = to_plain(v)
        return out

    # lists
    if isinstance(o, list):
        return [to_plain(x) for x in o]

    # generic objects with __dict__ (as a last resort)
    if hasattr(o, "__dict__") and not isinstance(o, type):
        cls = type(o)
        state = {k: to_plain(v) for k, v in vars(o).items()}
        return _tag(
            "__object__",
            {"module": cls.__module__, "qualname": cls.__qualname__, "state": state},
        )

    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")


# ---------- Decode back to live Python objects ----------

def from_plain(o: Any) -> Any:
    """
    Convert a JSON-deserialized structure back to Python objects.
    
    Reconstructs types, functions, dataclasses, etc. from tagged dictionaries.
    """
    if isinstance(o, list):
        return [from_plain(x) for x in o]

    if isinstance(o, dict):
        # tagged objects
        if "__py_type__" in o:
            return _decode_type(o["__py_type__"])
        if "__py_func__" in o:
            return _decode_func(o["__py_func__"])
        if "__enum__" in o:
            info = o["__enum__"]
            EnumT = _import_qualified(info["module"], info["qualname"])
            return EnumT(info["value"])
        if "__dataclass__" in o:
            info = o["__dataclass__"]
            cls = _import_qualified(info["module"], info["qualname"])
            fields = {k: from_plain(v) for k, v in info["fields"].items()}
            return cls(**fields)
        if "__ndarray__" in o:
            info = o["__ndarray__"]
            if np is None:
                return info["data"]  # fallback when numpy not available
            return np.array(info["data"], dtype=info.get("dtype"))
        if "__path__" in o:
            return Path(o["__path__"])
        if "__bytes__" in o:
            return base64.b64decode(o["__bytes__"])
        if "__complex__" in o:
            r, i = o["__complex__"]
            return complex(r, i)
        if "__set__" in o:
            return set(from_plain(x) for x in o["__set__"])
        if "__tuple__" in o:
            return tuple(from_plain(x) for x in o["__tuple__"])
        if "__partial__" in o:
            info = o["__partial__"]
            func = from_plain(info["func"])
            args = [from_plain(a) for a in info.get("args", [])]
            kwargs = {k: from_plain(v) for k, v in info.get("keywords", {}).items()}
            return functools.partial(func, *args, **kwargs)
        if "__object__" in o:
            info = o["__object__"]
            cls = _import_qualified(info["module"], info["qualname"])
            state = {k: from_plain(v) for k, v in info["state"].items()}
            # Try kwargs constructor first; fall back to setting attributes
            try:
                return cls(**state)
            except Exception:
                obj = cls.__new__(cls)
                for k, v in state.items():
                    setattr(obj, k, v)
                return obj

        # plain dict: recurse values
        return {k: from_plain(v) for k, v in o.items()}

    return o

def _locate_robots_dir(start: Optional[Path] = None) -> Optional[Path]:
    """
    Find the absolute path to dependencies/robots by walking up from this file.

    Returns
    -------
    Optional[Path]
        The resolved folder if found, else None.
    """
    base = (start or Path(__file__).resolve().parent)
    for root in [base, *base.parents]:
        candidate = root / "dependencies" / "robots"
        if candidate.is_dir():
            return candidate.resolve()
    return None

def _has_supported_suffix(s: str) -> bool:
    return s.lower().endswith((".usd", ".xml"))

def _rewrite_asset_paths_in_json(data: Any) -> Any:
    """
    Recursively find keys named 'asset_path' and, if the value starts with
    'robots/', rewrite it to an absolute path rooted at the project's
    dependencies/robots directory.

    Rules:
      - Only keys named 'asset_path' are considered.
      - Value must be a string ending with .usd/.USD or .xml.
      - If it starts with 'robots/' (or 'robots\\'), it's rewritten to:
            <abs_path_to_dependencies/robots>/<rest_of_the_path>
      - If dependencies/robots cannot be found, the value is left unchanged.
      - The function mutates the input structure in place and returns it.

    Examples:
      'robots/franka/franka.usd' ->
         '/abs/path/to/project/dependencies/robots/franka/franka.usd'

    Parameters
    ----------
    data : Any
        Arbitrary Python structure parsed from JSON (dicts/lists/scalars).

    Returns
    -------
    Any
        The same structure with paths rewritten where applicable.
    """
    robots_dir = _locate_robots_dir()
    if robots_dir is None:
        # Nothing to rewrite if we can't locate the robots folder.
        return data

    def rewrite_value(val: Any) -> Any:
        if not (isinstance(val, str) and _has_supported_suffix(val)):
            return val

        # Normalize separators for detection
        norm = val.replace("\\", "/")
        if norm.startswith("robots/"):
            rel = norm.split("/", 1)[1] if "/" in norm else ""
            target = (robots_dir / rel).resolve()

            # Always rewrite to an absolute path under dependencies/robots.
            # If the specific file doesn't exist, we still rewrite (env-independent path),
            # but you can gate this with `if target.exists():` if you prefer.
            return target.as_posix()

        return val

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "asset_path":
                    node[k] = rewrite_value(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item)
        # Scalars: nothing to do

    walk(data)
    return data

# ---------- Public API ----------

def save_cfg_json(cfg: Any, path: str) -> None:
    """
    Serialize configuration to JSON file (portable).
    
    Args:
        cfg: Configuration object to serialize
        path: Output file path
    """
    plain = to_plain(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=2, ensure_ascii=False)



def load_cfg_json(path: str) -> Any:
    """
    Load from JSON file and reconstruct original objects.
    
    Reconstructs dataclasses, functions, classes, etc.
    
    Args:
        path: Input file path
        
    Returns:
        Reconstructed configuration object
    """
    with open(path, "r", encoding="utf-8") as f:
        plain = json.load(f)

    plain = _rewrite_asset_paths_in_json(plain)
    reconstructed = from_plain(plain)
    return reconstructed

def dumps_cfg(cfg: Any) -> str:
    """
    Serialize configuration to JSON string.
    
    Args:
        cfg: Configuration object to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(to_plain(cfg), indent=2, ensure_ascii=False)


def loads_cfg(s: str) -> Any:
    """
    Deserialize from JSON string to reconstructed objects.
    
    Args:
        s: JSON string
        
    Returns:
        Reconstructed configuration object
    """
    plain = json.loads(s)
    plain = _rewrite_asset_paths_in_json(plain)
    reconstructed = from_plain(plain)
    return reconstructed
 
def load_plain_json(path: str) -> Any:
    """
    Load raw JSON data without reconstructing tagged objects.
    
    Useful for inspection in clean environments without dependencies.
    
    Args:
        path: Input file path
        
    Returns:
        Raw JSON data structure
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Diagnostics ----------

def find_nonserializable(o: Any, path: str = "root") -> None:
    """
    Attempt to serialize an object and print diagnostic information 
    for any non-serializable components.
    
    Args:
        o: Object to check for serializability
        path: Path label for diagnostic output (default: "root")
    """
    try:
        to_plain(o)
        return
    except TypeError:
        _walk_find(o, path)


def _walk_find(o: Any, path: str) -> None:
    """
    Recursively walk an object and report non-serializable components.
    
    Internal helper for find_nonserializable.
    """
    def report(msg: str) -> None:
        print(f"{path}: {msg}")

    # Check for serializable types
    if o is None or isinstance(o, (bool, int, float, str)):
        return

    if np is not None and (isinstance(o, np.generic) or isinstance(o, np.ndarray)):
        return

    if inspect.isfunction(o) or inspect.ismethod(o) or inspect.isbuiltin(o):
        report(f"function/callable -> {o}")
        return

    if isinstance(o, type):
        report(f"class/type -> {o}")
        return

    if isinstance(o, enum.Enum):
        return

    if dc.is_dataclass(o) and not isinstance(o, type):
        for f in dc.fields(o):
            _walk_find(getattr(o, f.name), f"{path}.{f.name}")
        return

    if isinstance(o, dict):
        for k, v in o.items():
            _walk_find(v, f"{path}[{k!r}]")
        return

    if isinstance(o, (list, tuple, set)):
        for i, x in enumerate(list(o) if isinstance(o, set) else o):
            _walk_find(x, f"{path}[{i}]")
        return

    if isinstance(o, (Path, bytes, bytearray, complex)):
        return

    if hasattr(o, "__dict__") and not isinstance(o, type):
        for k, v in vars(o).items():
            _walk_find(v, f"{path}.{k}")
        return

    report(f"unknown non-serializable type: {type(o).__name__} -> {o!r}")


# ---------- Interface tools ---------------------

_TAGS_WITH_MODULES: Tuple[str, ...] = (
    "__dataclass__", "__py_type__", "__py_func__", "__enum__", "__object__"
)


def _replace_first_component(module: str, old_first: str, new_first: str) -> str:
    """
    Replace the first component of a module path if it matches old_first.
    
    Args:
        module: Module path (e.g., "myapp.config.settings")
        old_first: First component to replace (e.g., "myapp")
        new_first: Replacement for first component (e.g., "yourapp")
        
    Returns:
        Modified module path
    """
    parts = module.split(".")
    if not parts:
        return module
    if parts[0] == old_first:
        parts[0] = new_first
    return ".".join(parts)


def _walk_and_rename_modules(d: Any, old_first: str, new_first: str) -> int:
    """
    Recursively walk data structure and rename module paths.
    
    Internal helper that returns count of replacements performed.
    """
    count = 0
    if isinstance(d, list):
        for x in d:
            count += _walk_and_rename_modules(x, old_first, new_first)
        return count
    if isinstance(d, dict):
        for tag in _TAGS_WITH_MODULES:
            if tag in d and isinstance(d[tag], dict) and "module" in d[tag]:
                mod_before = d[tag]["module"]
                mod_after = _replace_first_component(mod_before, old_first, new_first)
                if mod_after != mod_before:
                    d[tag]["module"] = mod_after
                    count += 1
        for v in d.values():
            count += _walk_and_rename_modules(v, old_first, new_first)
        return count
    return count


def rename_modules_in_data(data: Any, old_first: str, new_first: str) -> int:
    """
    In-place: replace the first package component 'old_first' -> 'new_first'
    inside module paths for all tagged objects in 'data'.
    
    Args:
        data: Data structure containing tagged objects
        old_first: First component to replace
        new_first: Replacement for first component
        
    Returns:
        Number of replacements performed
    """
    return _walk_and_rename_modules(data, old_first, new_first)


def rename_modules(
    in_path: str, 
    old_first: str, 
    new_first: str, 
    out_path: Optional[str] = None
) -> int:
    """
    Open JSON file, update module paths, and write out.
    
    If out_path is None, overwrite the input file.
    
    Args:
        in_path: Input JSON file path
        old_first: First component to replace
        new_first: Replacement for first component
        out_path: Output file path (default: overwrite input)
        
    Returns:
        Number of replacements performed
    """
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = rename_modules_in_data(data, old_first, new_first)
    target = out_path or in_path
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Renamed {n} module references: {old_first} -> {new_first}. Wrote: {target}")
    return n


def _collect_module_refs(
    d: Any, 
    only_first: Optional[str] = None
) -> Dict[str, Set[str]]:
    """
    Collect module references from tagged objects.
    
    Returns a dictionary mapping module paths to sets of qualified names.
    
    Args:
        d: Data structure to scan
        only_first: If provided, only include modules whose first component matches
        
    Returns:
        {module_path: set(qualnames)}
    """
    out: Dict[str, Set[str]] = {}
    
    def rec(o: Any) -> None:
        if isinstance(o, list):
            for x in o:
                rec(x)
            return
        if isinstance(o, dict):
            for tag in _TAGS_WITH_MODULES:
                if (tag in o and isinstance(o[tag], dict) and 
                    "module" in o[tag] and "qualname" in o[tag]):
                    mod = o[tag]["module"]
                    if only_first is None or mod.split(".", 1)[0] == only_first:
                        qn = o[tag]["qualname"]
                        out.setdefault(mod, set()).add(qn)
            for v in o.values():
                rec(v)
            return
    
    rec(d)
    return out


def _tree_insert(
    tree: dict, 
    path: List[str], 
    leaf_name: str, 
    items: Set[str]
) -> None:
    """
    Insert items into a tree structure at the specified path.
    
    Internal helper for building directory-like trees.
    """
    node = tree
    for p in path[:-1]:
        node = node.setdefault(p, {})
    leaf = path[-1] + ".py"
    node.setdefault(leaf, set()).update(items)


def _print_tree(node: dict, indent: int = 0) -> None:
    """
    Print a tree structure with indentation.
    
    Internal helper for displaying module reference trees.
    """
    pad = "  " * indent
    for k in sorted(node.keys()):
        v = node[k]
        if isinstance(v, dict):
            print(f"{pad}{k}/")
            _print_tree(v, indent + 1)
        else:
            # v is a set of qualnames
            qns = ", ".join(sorted(v))
            print(f"{pad}{k}  # {qns}")


def print_package_tree_from_data(data: Any, first_word: str) -> None:
    """
    Print a tree of referenced modules starting with the specified prefix.
    
    Suggests which files might need to be copied for portability.
    
    Example output:
    omni/
      isaac/
        lab/
          sim/
            utils.py  # RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
    
    Args:
        data: Data structure containing tagged objects
        first_word: Module prefix to filter by (e.g., "omni")
    """
    refs = _collect_module_refs(data, only_first=first_word)
    if not refs:
        print(f"No module references found starting with '{first_word}'.")
        return

    # Build a directory-like tree, converting module paths to file hints
    tree: dict = {}
    for mod, qns in refs.items():
        parts = mod.split(".")
        if parts[0] != first_word:
            continue
        if len(parts) == 1:
            # top package as __init__.py
            _tree_insert(tree, [parts[0], "__init__"], "__init__", qns)
        else:
            _tree_insert(tree, parts, parts[-1], qns)

    print(f"\nReferenced modules starting with '{first_word}':")
    _print_tree(tree)


def print_package_tree_from_json(json_path: str, first_word: str) -> None:
    """
    Load JSON file and print package tree for modules starting with specified prefix.
    
    Args:
        json_path: Path to JSON file
        first_word: Module prefix to filter by
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print_package_tree_from_data(data, first_word)


# ---------- Quick self-test (optional) ----------

if __name__ == "__main__":
    from dataclasses import dataclass

    def top_level_fn(x: int) -> int:
        """Test function for serialization."""
        return x + 1

    @dataclass
    class Child:
        """Test dataclass for serialization."""
        a: int
        f: Any

    @dataclass
    class Parent:
        """Test parent dataclass for serialization."""
        prim_path: str
        class_type: Any
        spawn: Child

    # Create test configuration
    cfg = Parent(
        prim_path="{ENV_REGEX_NS}/Robot",
        class_type=ValueError,   # a class/type
        spawn=Child(a=3, f=top_level_fn),  # a function
    )

    # Test serialization round-trip
    path = "cfg_test.json"
    save_cfg_json(cfg, path)
    restored = load_cfg_json(path)
    
    # Verify restoration
    assert type(restored) is Parent
    assert type(restored.spawn) is Child
    assert restored.spawn.f(41) == 42
    assert restored.class_type is ValueError
    print("Round-trip OK")

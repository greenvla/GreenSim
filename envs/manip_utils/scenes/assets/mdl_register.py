from pathlib import Path
from pxr import Usd, UsdShade, Sdf

_REGISTERED_MDL_ROOTS = {}


def _find_mdl_root_for_usd(usd_path: str) -> Path | None:
    usd_path = Path(usd_path).resolve()

    candidates = []
    for p in [usd_path.parent, *usd_path.parents]:
        candidates.extend([
            p / "Materials" / "2023_2_1",
            p / "isaac-sim-assets" / "Materials" / "2023_2_1",
            p / "Isaac Sim 4.5" / "isaac-sim-assets" / "Materials" / "2023_2_1",
            p / "Materials",
        ])

    seen = set()
    uniq = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq.append(c)

    for c in uniq:
        if c.exists() and c.is_dir():
            return c

    return None


def _register_mdl_root(mdl_root: str) -> str:
    import omni.mdl.neuraylib

    mdl_root = str(Path(mdl_root).resolve())
    if mdl_root in _REGISTERED_MDL_ROOTS:
        return _REGISTERED_MDL_ROOTS[mdl_root]

    omni.mdl.neuraylib.ensure_running()

    ext_name = f"runtime_mdl_{abs(hash(mdl_root))}"
    omni.mdl.neuraylib.register_extension_content(ext_name, mdl_root)
    _REGISTERED_MDL_ROOTS[mdl_root] = ext_name
    print(f"[MDL] registered root: {mdl_root}")
    return ext_name


def _rewrite_usd_mdl_paths_to_package_paths(source_usd_path: str) -> str:
    """
    Makes a runtime copy of the USD and rewrites the mdl sourceAsset:
    absolute file path -> package-relative path relative to the found MDL root.
    """
    source_usd_path = str(Path(source_usd_path).resolve())
    mdl_root = _find_mdl_root_for_usd(source_usd_path)
    if mdl_root is None:
        return source_usd_path

    _register_mdl_root(str(mdl_root))

    src = Path(source_usd_path)
    out_path = src.with_name(src.stem + "__mdl_fixed.usd")

    stage = Usd.Stage.Open(source_usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {source_usd_path}")

    stage.Export(str(out_path))

    out_stage = Usd.Stage.Open(str(out_path))
    if out_stage is None:
        raise RuntimeError(f"Failed to open rewritten USD: {out_path}")

    changed = 0

    for prim in out_stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue

        shader = UsdShade.Shader(prim)
        impl_src = shader.GetImplementationSourceAttr().Get()
        if impl_src != UsdShade.Tokens.sourceAsset:
            continue

        source_asset = shader.GetSourceAsset("mdl")
        if not source_asset:
            continue

        asset_path = source_asset.path if hasattr(source_asset, "path") else str(source_asset)
        if not asset_path or not str(asset_path).endswith(".mdl"):
            continue

        asset_path = str(asset_path)

        # absolute file path -> package-relative path
        try:
            abs_asset = Path(asset_path).resolve()
        except Exception:
            continue

        try:
            rel_path = abs_asset.relative_to(mdl_root.resolve()).as_posix()
        except Exception:
            continue

        shader.SetSourceAsset(Sdf.AssetPath(rel_path), "mdl")
        changed += 1

    out_stage.GetRootLayer().Save()
    print(f"[MDL] rewritten USD saved: {out_path}, changed shaders: {changed}")
    return str(out_path)

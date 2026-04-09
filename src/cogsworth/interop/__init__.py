def __getattr__(name):
    import importlib
    if name in ("compas",):
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'cogsworth.interop' has no attribute {name!r}")

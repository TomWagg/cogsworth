def __getattr__(name):
    import importlib
    if name in ("mist", "observables"):
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'cogsworth.obs' has no attribute {name!r}")

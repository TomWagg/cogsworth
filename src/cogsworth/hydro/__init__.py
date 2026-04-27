def __getattr__(name):
    import importlib
    if name in ("pop", "potential", "rewind", "utils"):
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'cogsworth.hydro' has no attribute {name!r}")

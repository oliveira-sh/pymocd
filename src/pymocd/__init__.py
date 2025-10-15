def hello_world():
    return "Hello Python frontend!"


def _get_rust_exports():
    try:
        from . import pymocd as _rust_module
        
        rust_exports = [name for name in dir(_rust_module) 
                       if not name.startswith('_')]
        return rust_exports
    except ImportError:
        return []

try:
    from . import pymocd as _rust_module
    
    rust_exports = _get_rust_exports()
    for name in rust_exports:
        globals()[name] = getattr(_rust_module, name)
    
    __all__ = ['hello_world'] + rust_exports
    
except ImportError:
    __all__ = ['hello_world']
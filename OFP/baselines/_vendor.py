"""Load vendored layers without leaving generic module names in sys.modules."""
import importlib.util
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'third_party'

def load_layers(root,files):
    prefixes={name.split('.')[0] for name in files}
    saved={k:v for k,v in sys.modules.items() if k.split('.')[0] in prefixes}
    for key in saved: del sys.modules[key]
    loaded={}
    try:
        for prefix in prefixes:
            package=types.ModuleType(prefix)
            package.__path__=[str(root/prefix)]
            sys.modules[prefix]=package
        for name in files:
            spec=importlib.util.spec_from_file_location(name,root/(name.replace('.','/')+'.py'))
            module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            spec.loader.exec_module(module)
            loaded[name]=module
        return loaded
    finally:
        for key in list(sys.modules):
            if key.split('.')[0] in prefixes: del sys.modules[key]
        sys.modules.update(saved)

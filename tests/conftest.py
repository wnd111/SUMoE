from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
while str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))
for module_name, module in list(sys.modules.items()):
    if module_name == "sumoe" or module_name.startswith("sumoe."):
        module_file = getattr(module, "__file__", "") or ""
        if str(SRC) not in module_file:
            del sys.modules[module_name]

from __future__ import annotations

from pathlib import Path

import gmsh


gmsh_module = Path(gmsh.__file__).resolve()
module_directory = gmsh_module.parent
candidates = []
for root in (module_directory, module_directory.parent, module_directory.parent.parent):
    candidates.extend(root / name for name in (getattr(gmsh, "libname", ""), "gmsh.dll") if name)
library = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
if library is None:
    raise RuntimeError(f"Unable to locate the Gmsh runtime DLL near {gmsh_module}")

binaries = [(str(library), ".")]
hiddenimports = []
datas = []

from __future__ import annotations

if __name__ == "profile":
    import importlib.util
    import sysconfig
    from pathlib import Path

    standard_path = Path(sysconfig.get_path("stdlib")) / "profile.py"
    specification = importlib.util.spec_from_file_location("_stdlib_profile", standard_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load Python standard-library profile module: {standard_path}")
    standard_profile = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(standard_profile)
    run = standard_profile.run
    runctx = standard_profile.runctx
    Profile = standard_profile.Profile
    _Utils = standard_profile._Utils
else:
    from profile_efficiency import main

    if __name__ == "__main__":
        main()

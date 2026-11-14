"""Compile the local Gazebo 10 TT motor system and report its plugin directory."""

from pathlib import Path
import os
import subprocess

HERE = Path(__file__).resolve().parent
BUILD = HERE / "plugins/build"


def ensure_plugin():
    sources = [HERE / "plugins/TTMotorSystem.cc", HERE / "plugins/CMakeLists.txt"]
    libraries = list(BUILD.glob("*ugv_tt_motor.*")) if BUILD.exists() else []
    libraries = [p for p in libraries if p.suffix in (".dylib", ".so", ".dll")]
    newest = max(p.stat().st_mtime for p in sources)
    if not libraries or max(p.stat().st_mtime for p in libraries) < newest:
        env = dict(os.environ)
        prefixes = [p for p in ("/opt/homebrew", "/usr/local") if Path(p).exists()]
        options = []
        # Relocated Ubuntu Harmonic packages (non-Debian host): Debian multiarch
        # CMake configs are not on the default search path, and the plugin must
        # link the prefix's tinyxml2 rather than a different host soname.
        harmonic = Path(env.get("UGV_GZ_PREFIX", "/opt/gz-harmonic/usr"))
        multiarch = harmonic / "lib/x86_64-linux-gnu"
        if (multiarch / "cmake/gz-sim8").exists():
            prefixes = [str(multiarch / "cmake"), str(harmonic)] + prefixes
            tinyxml = multiarch / "libtinyxml2.so.10"
            if tinyxml.exists():
                options += [
                    f"-DTINYXML2_LIBRARY_tinyxml2={tinyxml}",
                    f"-Dpkgcfg_lib_TINYXML2_tinyxml2={tinyxml}",
                ]
        env["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefixes + [env.get("CMAKE_PREFIX_PATH", "")])
        subprocess.run(
            [
                "cmake",
                "-S",
                str(HERE / "plugins"),
                "-B",
                str(BUILD),
                "-DCMAKE_BUILD_TYPE=Release",
                *options,
            ],
            check=True,
            env=env,
        )
        subprocess.run(["cmake", "--build", str(BUILD), "-j", "4"], check=True, env=env)
    return BUILD


def plugin_environment():
    directory = ensure_plugin()
    env = dict(os.environ)
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = (
        str(directory) + os.pathsep + env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    )
    return env


if __name__ == "__main__":
    print(ensure_plugin())

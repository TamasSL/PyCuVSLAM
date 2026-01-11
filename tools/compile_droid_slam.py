import shutil
from pathlib import Path

from bowler import Query


def move_files_in_droid_slam_api(project_root: Path):
    code_dir = project_root / "droid_slam"
    target_dir = project_root / "droid_slam_api"

    if not code_dir.exists():
        if target_dir.exists():
            print("=> 1) Droid SLAM API already exists, skipping.")
            return target_dir
        raise ValueError("Droid SLAM code doesn't exist")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(code_dir), str(target_dir))
    print("=> 1) Code successfully moved to Droid SLAM API.")

    return target_dir


def update_droid_slam_imports(code_dir: Path):
    file_containing_import = code_dir / "droid_slam" / "droid.py"
    with open(file_containing_import) as f:
        for line in f.readlines():
            if "from droid_slam" in line:
                print("=> 2) Imports from droid_slam already exists, skipping.")
                return

    query = Query(str(code_dir))
    modules = [path.stem for path in (code_dir / "droid_slam").iterdir()]
    modules_mapping = {key: f"droid_slam.{key}" for key in modules}

    for old, new in modules_mapping.items():
        query = query.select_module(old).rename(new)

    query.write()
    print("=> 2) Updated imports to be relative to droid_slam.")


def move_python_files_in_droid_slam_api(project_root: Path, code_dir: Path):
    files = ["demo.py", "train.py", "view_reconstruction.py"]
    moved = False
    for file in files:
        if (project_root / file).exists():
            shutil.move(str(project_root / file), str(code_dir))
            moved = True

    if not moved:
        print("=> 3) Python files already moved to Droid SLAM API, skipping.")
    else:
        print("=> 3) Python files moved to Droid SLAM API.")


toml_file_content = """[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "droid-slam"
version = "0.0.0"
description = "DROID-SLAM Python API"

[tool.setuptools.packages.find]
where = ["."]
include = ["droid_slam", "droid_slam.*"]
"""


def add_pyproject_toml(code_dir: Path):
    file_path = code_dir / "pyproject.toml"
    if file_path.exists():
        print("=> 4) TOML file already existing, skipping.")
        return
    with open(file_path, "w") as f:
        f.write(toml_file_content)
    print("=> 4) Succesfully written the TOML file")


def main():
    project_root = Path(__file__).resolve().parent.parent / "thirdparty" / "droid_slam"
    code_dir = move_files_in_droid_slam_api(project_root)
    update_droid_slam_imports(code_dir)
    move_python_files_in_droid_slam_api(project_root, code_dir)
    add_pyproject_toml(code_dir)


if __name__ == "__main__":
    main()

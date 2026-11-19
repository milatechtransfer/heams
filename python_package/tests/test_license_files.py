"""Each Python distribution ships its own copy of the root LICENSE and NOTICE.

Setuptools only finds license files inside a project's own directory, so
``python_package/`` and every ``memory_plugins/*/`` keep copies of the root
files. This test fails when a copy drifts from the root.
"""

import pytest

from tests.paths import PACKAGE_ROOT, REPO_ROOT

PROJECT_DIRS = [PACKAGE_ROOT, *sorted((REPO_ROOT / "memory_plugins").glob("*/"))]


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
@pytest.mark.parametrize("project_dir", PROJECT_DIRS, ids=lambda path: path.name)
def test_license_file_matches_root(project_dir, name) -> None:
    copy = project_dir / name

    assert copy.is_file(), f"{copy} is missing; copy it from the repo root"
    assert copy.read_text() == (REPO_ROOT / name).read_text(), (
        f"{copy} differs from the repo root {name}; copy it again"
    )

# AuToGraFS Code Review and Improvements

This document compiles a comprehensive review of the AuToGraFS codebase, identifying potential bugs, improvements, and general advice for maintainability and robustness.

---

## Table of Contents

1. [Critical Bugs](#critical-bugs)
2. [Code Quality Issues](#code-quality-issues)
3. [Performance Improvements](#performance-improvements)
4. [API Design Improvements](#api-design-improvements)
5. [Documentation Issues](#documentation-issues)
6. [Testing Gaps](#testing-gaps)
7. [Type Safety & Static Analysis](#type-safety--static-analysis)
8. [Miscellaneous Recommendations](#miscellaneous-recommendations)

---

## Critical Bugs

### 1. Class Attributes Shared Across Instances (`builder.py`)

**Location:** `builder.py`, lines 43-44

```python
class Autografs(object):
    topologies: dict[str, Topology] = {}
    sbu: dict[str, Fragment] = {}
```

**Issue:** Class attributes `topologies` and `sbu` are mutable dictionaries defined at the class level. This means all instances of `Autografs` share the same dictionaries, leading to unexpected data sharing between instances.

**Fix:** Move these to instance attributes in `__init__`:

```python
def __init__(self, xyzfile=None, topofile=None):
    self.topologies: dict[str, Topology] = {}
    self.sbu: dict[str, Fragment] = {}
    # ... rest of init
```

---

### 2. `functools.cached_property` Invalidation Issue (`structure.py`)

**Location:** `structure.py`, `Fragment.max_dummy_distance`

**Issue:** The `max_dummy_distance` property is cached using `@functools.cached_property`, but methods like `rotate()` modify the atomic coordinates. After rotation, the cached value becomes stale.

**Fix:** Either:
- Clear the cache when coordinates change, or
- Use a regular property (performance trade-off), or
- Implement `__setattr__` to invalidate the cache when `atoms` is modified

```python
def rotate(self, theta: float) -> None:
    # ... rotation logic ...
    # Invalidate cache
    if "max_dummy_distance" in self.__dict__:
        del self.__dict__["max_dummy_distance"]
```

---

### 3. Silent Warning Suppression (`builder.py`, `structure.py`, `utils.py`)

**Location:** Multiple files with `warnings.filterwarnings("ignore")`

**Issue:** Global warning suppression can hide important deprecation warnings or runtime issues, making debugging difficult.

**Fix:** Use context managers for specific operations or filter only expected warnings:

```python
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen")
    # specific operation
```

---

### 4. Potential Division by Zero (`builder.py`)

**Location:** `builder.py`, `_align_all_mappings` method

```python
score = slot_weight * rmse / (len(slot.atoms) * len(topology.mappings[slot]))
```

**Issue:** If `len(slot.atoms)` or `len(topology.mappings[slot])` is 0, this will raise a `ZeroDivisionError`.

**Fix:** Add defensive checks:

```python
denominator = len(slot.atoms) * len(topology.mappings[slot])
if denominator == 0:
    logger.warning(f"Empty slot encountered: {slot}")
    score = float('inf')
else:
    score = slot_weight * rmse / denominator
```

---

### 5. Missing Return Type in `_validate_mappings` (`builder.py`)

**Location:** `builder.py`, `_validate_mappings` method

**Issue:** The function modifies `mappings` in place and also creates `true_mappings`, but the return type annotation says `dict[int, Fragment]` while technically returning the modified version. This is confusing.

**Fix:** Don't modify the input parameter; create a new dictionary from the start:

```python
def _validate_mappings(
    self, topology: Topology, mappings: dict[Fragment | int, Fragment | str]
) -> dict[int, Fragment]:
    validated = {}
    for k in topology.mappings.keys():
        assert k in mappings, f"Unfilled {k} slot."
    # Create new dict without modifying input
    # ...
```

---

## Code Quality Issues

### 6. Use of `object` Base Class

**Location:** `builder.py`, `structure.py`

```python
class Autografs(object):
class Fragment(object):
class Topology(object):
```

**Issue:** In Python 3, explicit inheritance from `object` is unnecessary.

**Fix:** Simply use:

```python
class Autografs:
class Fragment:
class Topology:
```

---

### 7. Inconsistent Return Statements (`builder.py`, `structure.py`)

**Location:** Multiple methods ending with `return None`

```python
def _setup_sbu(self, xyzfile: str | None = None) -> None:
    # ... logic ...
    return None  # Unnecessary
```

**Issue:** Explicit `return None` at the end of `None`-returning functions is redundant.

**Fix:** Remove unnecessary `return None` statements or be consistent throughout the codebase.

---

### 8. Magic Numbers and Hardcoded Values

**Location:** Various files

- `builder.py`: `Ns=3` in `brute()` call
- `builder.py`: `abc_norm * 0.1` and `abc_norm * 2.0` bounds
- `utils.py`: `tol=0.3`, `cutoff=10.0` in `EconNN`
- `structure.py`: `tolerance=0.1` in `PointGroupAnalyzer`

**Fix:** Define named constants:

```python
# builder.py
BRUTE_SEARCH_GRID_POINTS = 3
SCALE_SEARCH_MIN_FACTOR = 0.1
SCALE_SEARCH_MAX_FACTOR = 2.0

# utils.py
BOND_TOLERANCE = 0.3
BOND_CUTOFF = 10.0
```

---

### 9. Bare `except` Clauses

**Location:** `structure.py`, `has_compatible_symmetry` method

```python
except Exception:
    return False
```

**Issue:** Catching all exceptions hides potential bugs. Also catches `KeyboardInterrupt`, `SystemExit`, etc. if using bare `except`.

**Fix:** Catch specific exceptions:

```python
except (ValueError, AttributeError) as e:
    logger.debug(f"Symmetry compatibility check failed: {e}")
    return False
```

---

### 10. Commented-Out Code (`builder.py`)

**Location:** `builder.py`, bottom of file and around `try/except` blocks

**Issue:** Large blocks of commented-out code in the `if __name__ == "__main__"` section and commented `try/except` blocks reduce readability.

**Fix:** Remove commented code or move to separate example files.

---

## Performance Improvements

### 11. Inefficient Nested Loops for Distance Calculation (`structure.py`)

**Location:** `structure.py`, `max_dummy_distance` property

```python
for i in range(len(dummies)):
    c_i = dummies[i]
    for j in range(i, len(dummies)):
        c_j = dummies[j]
        dist = max(dist, float(np.linalg.norm(c_i - c_j)))
```

**Fix:** Use scipy's pdist for vectorized computation:

```python
from scipy.spatial.distance import pdist

@functools.cached_property
def max_dummy_distance(self) -> float:
    dummies = self.extract_dummies().cart_coords
    if len(dummies) < 2:
        return 0.0
    return float(pdist(dummies).max())
```

---

### 12. Repeated File I/O in `load_uff_lib` (`utils.py`)

**Location:** `utils.py`, `load_uff_lib` function

**Issue:** The UFF library CSV is read every time `load_uff_lib` is called.

**Fix:** Cache the full library at module level:

```python
_UFF_LIBRARY_CACHE = None

def load_uff_lib(mol: Molecule) -> tuple[pandas.DataFrame, list[str]]:
    global _UFF_LIBRARY_CACHE
    if _UFF_LIBRARY_CACHE is None:
        path = os.path.join(autografs.data.__path__[0], "uff4mof.csv")
        _UFF_LIBRARY_CACHE = pandas.read_csv(path)
    # ... rest of function using _UFF_LIBRARY_CACHE
```

---

### 13. Inefficient String Building (`utils.py`)

**Location:** `utils.py`, `networkx_to_gulp` function

```python
out_string += "..."  # Repeated string concatenation
```

**Fix:** Use a list and join at the end:

```python
def networkx_to_gulp(...) -> str:
    lines = []
    lines.append("opti conp molmec noautobond cartesian noelectrostatics ocell")
    # ...
    return "\n".join(lines)
```

---

### 14. Deep Copy in Hot Path (`builder.py`)

**Location:** `builder.py`, `_validate_mappings` and `build` methods

**Issue:** `copy.deepcopy()` is called frequently, which is slow for complex objects.

**Fix:** Consider implementing custom `__copy__` and `__deepcopy__` methods, or use shallow copies where appropriate.

---

## API Design Improvements

### 15. Inconsistent Method Naming

**Location:** Various

- `list_topologies()` vs `list_building_units()` - inconsistent pluralization style
- `_setup_sbu()` vs `_setup_topologies()` - one uses abbreviation, one doesn't

**Fix:** Standardize naming:
- `list_topologies()` / `list_sbus()` or
- `list_topology_names()` / `list_sbu_names()`

---

### 16. Missing `__slots__` for Performance-Critical Classes

**Location:** `structure.py`

**Fix:** Add `__slots__` to `Fragment` and `Topology` for memory efficiency:

```python
class Fragment:
    __slots__ = ('atoms', 'symmetry', 'name')
    # ...
```

Note: This would require removing `@functools.cached_property` or handling it differently.

---

### 17. `flip()` Method Raises NotImplementedError

**Location:** `structure.py`, `Fragment.flip` method

**Issue:** The README shows usage of `mof.flip(index=8)`, but the underlying method raises `NotImplementedError`.

**Fix:** Either implement the method or remove it from documentation.

---

### 18. README API Mismatch

**Location:** `README.rst`

**Issue:** The README shows methods like `mofgen.make()`, `mofgen.set_topology()`, `mof.write()`, `mof.view()`, `mof.get_supercell()`, etc. that don't exist in the current codebase.

**Fix:** Update README to match actual API, or implement the documented features.

---

### 19. Return Type Annotation Inconsistency

**Location:** `builder.py`, `list_building_units` method

```python
def list_building_units(...) -> dict[str, Fragment]:
```

**Issue:** Actually returns `dict[str, list[str]]` (SBU names, not Fragment objects).

**Fix:** Correct the return type annotation:

```python
def list_building_units(...) -> dict[Fragment, list[str]]:
```

---

## Documentation Issues

### 20. Outdated Docstrings

**Location:** Multiple files

**Issue:** Some docstrings reference parameters or behaviors that have changed.

**Examples:**
- `builder.py`: `mappings` parameter description doesn't fully explain Fragment key usage
- `structure.py`: `scale_slots` TODO comment mentions renaming that was never done

---

### 21. Missing Module-Level Examples

**Location:** `utils.py`

**Issue:** The module docstring lists functions but doesn't provide a quick usage example.

**Fix:** Add a brief example:

```python
"""
Utility functions for AuToGraFS framework generation.

Example
-------
>>> from autografs.utils import xyz_to_sbu
>>> sbus = xyz_to_sbu("my_sbus.xyz")
>>> print(list(sbus.keys()))
"""
```

---

### 22. `data/__init__.py` is Empty

**Location:** `src/autografs/data/__init__.py`

**Issue:** Empty `__init__.py` files should at least contain a docstring describing the data module.

**Fix:**

```python
"""
Data files for AuToGraFS.

This package contains:
- defaults.xyz: Default Secondary Building Units
- topologies.pkl: Pre-processed RCSR topologies
- uff4mof.csv: UFF force field parameters
- covalent_radii.csv: Covalent radii data
"""
```

---

## Testing Gaps

### 23. Tests Requiring External Data Files

**Location:** `tests/test_builder.py`

**Issue:** Many tests are skipped with `@requires_data` marker because they depend on `topologies.pkl` which may not exist.

**Fix:** 
- Include a minimal test topology file in the test fixtures
- Or mock the topology loading

---

### 24. No Tests for Error Conditions

**Location:** `tests/`

**Issue:** Few tests verify error handling, such as:
- Invalid XYZ file format
- Missing topology files
- Incompatible SBU-slot combinations

**Fix:** Add negative test cases:

```python
def test_xyz_to_sbu_invalid_file():
    with pytest.raises(FileNotFoundError):
        utils.xyz_to_sbu("nonexistent.xyz")

def test_xyz_to_sbu_malformed():
    with tempfile.NamedTemporaryFile(...) as f:
        f.write("invalid content")
        with pytest.raises(ValueError):
            utils.xyz_to_sbu(f.name)
```

---

### 25. No Integration Tests for `build_all`

**Location:** `tests/test_builder.py`

**Issue:** The `build_all` method is complex but has no dedicated tests.

---

### 26. Test File Cleanup

**Location:** `tests/test_utils.py`

**Issue:** Tests that write files (like `networkx_to_gulp`) need cleanup in case of test failure.

**Fix:** Use `pytest.fixture` with cleanup or context managers consistently.

---

## Type Safety & Static Analysis

### 27. Union Types Could Use `|` Syntax Consistently

**Location:** Various

The codebase already uses Python 3.13 with `from __future__ import annotations`, but some type hints could be cleaner:

```python
# Current
def __init__(self, name: str, slots: list[Fragment], cell: np.ndarray | Lattice) -> None:

# Could also accept
def __init__(self, name: str, slots: list[Fragment], cell: np.ndarray | Lattice) -> None:
```

This is already done correctly in most places.

---

### 28. Missing `py.typed` Marker Note

**Location:** `src/autografs/py.typed`

**Good:** The `py.typed` marker exists, indicating the package supports type checking. However, ensure all public APIs have complete type annotations.

---

### 29. NDArray Type Annotations

**Location:** `structure.py`

```python
from numpy.typing import NDArray
```

**Issue:** `NDArray` is imported but not used extensively. Consider using it for array return types:

```python
def extract_dummies(self) -> NDArray[np.floating]:
    # ...
```

---

## Miscellaneous Recommendations

### 30. Use `pathlib` Instead of `os.path`

**Location:** Various files

**Current:**
```python
import os
path = os.path.join(autografs.data.__path__[0], "defaults.xyz")
```

**Recommended:**
```python
from pathlib import Path
path = Path(autografs.data.__path__[0]) / "defaults.xyz"
```

---

### 31. Logging Configuration in `__init__.py`

**Location:** `src/autografs/__init__.py`

```python
logging.basicConfig(
    format="[AuToGraFS] %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)
```

**Issue:** Libraries shouldn't call `basicConfig()` as it can override the application's logging configuration.

**Fix:** Only add the `NullHandler`:

```python
logging.getLogger(__name__).addHandler(logging.NullHandler())
# Remove basicConfig call - let the application configure logging
```

---

### 32. Add `__all__` to Submodules

**Location:** `builder.py`, `structure.py`, `utils.py`

**Issue:** Submodules lack `__all__`, making it unclear what the public API is.

**Fix:**

```python
# builder.py
__all__ = ["Autografs"]

# structure.py
__all__ = ["Fragment", "Topology"]

# utils.py
__all__ = [
    "format_mappings",
    "get_xyz_names",
    "xyz_to_sbu",
    # ...
]
```

---

### 33. Consider Using `dataclasses` for Simple Data Containers

**Location:** `structure.py`

The `Fragment` and `Topology` classes could potentially benefit from `@dataclass` decoration, though they have enough custom behavior that it may not be worth it.

---

### 34. `tqdm.auto` Import

**Location:** `builder.py`

```python
from tqdm.auto import tqdm
```

**Good:** Using `tqdm.auto` for automatic notebook/terminal detection is a good practice.

---

### 35. Version Synchronization

**Location:** `__init__.py` and `pyproject.toml`

**Issue:** Version is defined in two places:
- `__init__.py`: `__version__ = "3.0.0"`
- `pyproject.toml`: `version = "3.0.0"`

**Fix:** Use dynamic versioning or a single source of truth:

```toml
# pyproject.toml
[tool.setuptools.dynamic]
version = {attr = "autografs.__version__"}
```

---

### 36. CI Badge Points to `master` Branch

**Location:** `README.rst`

```rst
.. |codecov| image:: https://codecov.io/gh/DCoupry/autografs/branch/master/graph/badge.svg
```

**Issue:** Badge points to `master` branch, but development is happening on feature branches.

---

### 37. Add Pre-commit Hooks Configuration

**Recommendation:** Add `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
```

---

## Summary

### Priority Fixes (Critical)

1. **Fix class-level mutable attributes** in `Autografs`
2. **Handle cached property invalidation** after coordinate modifications
3. **Fix division by zero** potential in alignment scoring
4. **Update README** to match actual API

### Quick Wins (Low Effort, High Value)

1. Remove explicit `object` inheritance
2. Add `__all__` to submodules
3. Remove logging `basicConfig()` call
4. Add docstring to `data/__init__.py`
5. Fix return type annotation on `list_building_units`

### Technical Debt Reduction

1. Extract magic numbers to named constants
2. Replace string concatenation with list joining
3. Cache UFF library at module level
4. Use `pathlib` consistently
5. Add more comprehensive test coverage

---

*Generated: November 2025*
*Branch: feature/unit-tests-issue-11*

# Test suite

The suite is divided by the kind of guarantee a test provides:

```text
tests/
  unit/          Isolated tests for data, metrics, methods, and GCN components
  equivalence/   Cross-method comparisons against a shared outcome contract
  integration/   Small end-to-end workflows spanning multiple modules
  fixtures/      Tiny static input files used by more than one test module
  helpers/       Reusable assertions and test-only utilities
  conftest.py    Deterministic graph, ranking, and random-number fixtures
```

Keep unit tests fast and independent of the real datasets. Equivalence tests
should state what equivalence means: identical order, the same top-k gene set,
or metrics equal within a documented numerical tolerance. Integration tests may
compose modules, but should still use synthetic inputs unless a test is
explicitly marked `slow`.

## Running tests

```bash
pytest
pytest -m unit
pytest -m equivalence
pytest -m integration
pytest -m "not slow"
```

Tests are automatically marked `unit`, `equivalence`, or `integration` based on
their containing directory. Use `@pytest.mark.slow` explicitly when needed.

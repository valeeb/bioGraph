# Integration tests

Exercise workflows spanning loading, splitting, scoring, ranking, and metrics.
Use the synthetic fixtures by default. Tests requiring PyTorch or substantial
runtime should use `pytest.importorskip("torch")` and `@pytest.mark.slow`.

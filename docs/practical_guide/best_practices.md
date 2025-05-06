## Best Practices

1. Always specify critical parameters in YAML files rather than relying on command-line overrides for better reproducibility.
2. Use environment-specific YAML files for parameters that are consistent across runs for that environment.
3. Use command-line overrides for experimental variations or one-off changes.
4. Document any non-standard parameter combinations in experiment logs.
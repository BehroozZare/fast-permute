# Creating Homa Python 

Create or update the development environment:

```
conda env update -f environment.yml
conda activate homa
```

All commands below assume the `homa` environment is active. 

The default Python build uses the settings in `pyproject.toml`: Python bindings on, examples/tests off, and CHOLMOD, MKL, cuDSS, and the Lloyd patcher enabled. Because cuDSS is enabled by default, the default build needs a working CUDA toolkit setup.


Build and install the package into the active environment:

```
python -m pip install -v . --no-build-isolation
```

After changing native code such as `python/src/homa_bindings.cpp`, CMake files, or C++ sources, force a rebuild and reinstall:

```
python -m pip install -v . --no-build-isolation --force-reinstall --no-deps
```

For editable pure-Python development:

```
python -m pip install -v -e . --no-build-isolation
```

### CPU-Only Build

To build without CUDA/cuDSS, disable both CUDA and cuDSS through pip's CMake config settings:

```
python -m pip install -v . --no-build-isolation `
  -Ccmake.define.HOMA_WITH_CUDA=OFF `
  -Ccmake.define.HOMA_WITH_CUDSS=OFF
```

Use the same flags with `--force-reinstall --no-deps` when switching an existing install from the default CUDA/cuDSS build to CPU-only.

## Build a Local Wheel

Build a wheel into `dist/` without installing it:

```
python -m pip wheel . -w dist --no-build-isolation
```

## Testing

Check that the package is installed and see which solver backends were built:

```
python -m pip show homa
python -c "import homa; print(homa.has_cholmod(), homa.has_mkl(), homa.has_cudss(), homa.has_cuda())"
```

And run the Python tests:

```
python -m pytest tests\python
```

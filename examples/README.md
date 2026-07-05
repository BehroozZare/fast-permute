## Building the examples

When examples are enabled, a solver example is built per enabled backend.
Disable the backends you do not have installed.

```bash
# CHOLMOD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_CHOLMOD=ON
cmake --build build

# MKL PARDISO
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_MKL=ON
cmake --build build

# cuDSS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_CUDSS=ON
cmake --build build
```

### Running the examples

```bash
./build/bin/cholmod_example -i input/meshes/dragon.obj
./build/bin/mkl_example     -i input/meshes/dragon.obj
./build/bin/cudss_example   -i input/meshes/dragon.obj
```

Optional flag: `-p <patch_size>` (default 512).

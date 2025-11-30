
#!/usr/bin/env bash

cd /home/behrooz/Desktop/Last_Project/gpu_ordering/build/benchmark

./gpu_ordering_benchmark \
  -i /media/behrooz/FarazHard/Last_Project/MIT_meshes/nefertiti.obj \
  -s CHOLMOD \
  -a POC_ND \
  -g 0
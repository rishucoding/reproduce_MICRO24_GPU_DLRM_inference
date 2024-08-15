# Paper: Pushing the Performance Envelope of DNN-based Recommendation Systems Inference on GPUs

# reproduce_MICRO24_GPU_DLRM_inference
Sharing the codebase and steps for artifact evaluation/reproduction for MICRO 2024 paper

## Steps to build and recompile pytorch stack

### Conda setup
* Download conda from https://www.anaconda.com/download/
* Install conda using "bash Anaconda-latest-Linux-x86_64.sh"
* Carefully set the installation path (PREFIX_)while installing on the terminal
* Do "which conda" to validate the path of the conda installation.

### PyTorch setup
```
#NOTE: check the env_files directory for req.txt, pip_req_after_conda.txt, and CMakeLists.txt
conda create -y --name ae_dlrm_gpu --file env_files/req.txt
conda activate ae_dlrm_gpu
pip install -r env_files/pip_req_after_conda.txt 
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
git clone --recursive -b v2.1.0 https://github.com/pytorch/pytorch
cd pytorch
cp ../env_files/CMakeLists.txt ./
cp ../opt_designs/baseline/EmbeddingBag_timed.cu aten/src/ATen/native/cuda/EmbeddingBag.cu
export TORCH_CUDA_ARCH_LIST="8.0"
REL_WITH_DEB_INFO=1 USE_NATIVE_ARCH=1 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" USE_CUDA=1 USE_CUDNN=1 USE_NUMPY=1 python setup.py install
python setup.py develop && python -c "import torch"
#Verify pytorch is installed successfully by [python -c "import torch; print(torch.__version__)"]
```

### To run the baseline inference on GPU
```
cd scripts
python save_emb.py
bash quick.sh > dummy; grep 'Latency of Embedding Bag kernel' dummy |  cut -d' ' -f6 | cut -d'm' -f1 > log; python parser_Kernel_ET.py ;
```

### To evaluate OptMT inference on GPU
```
cd pytorch
python setup.py clean
export TORCH_CUDA_ARCH_LIST="8.0"
REL_WITH_DEB_INFO=1 USE_NATIVE_ARCH=1 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" USE_CUDA=1 USE_CUDNN=1 USE_NUMPY=1 CMAKE_CUDA_FLAGS="-maxrregcount 42" python setup.py install
cd ../scripts
bash quick.sh > dummy; grep 'Latency of Embedding Bag kernel' dummy |  cut -d' ' -f6 | cut -d'm' -f1 > log; python parser_Kernel_ET.py ;
```

### To evaluate RPF + OptMT inference on GPU
```
cd ../pytorch
cp ../opt_designs/prefetching/EmbeddingBag_regpref.txt aten/src/ATen/native/cuda/EmbeddingBag.cu
export TORCH_CUDA_ARCH_LIST="8.0"
REL_WITH_DEB_INFO=1 USE_NATIVE_ARCH=1 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" USE_CUDA=1 USE_CUDNN=1 USE_NUMPY=1 CMAKE_CUDA_FLAGS="-maxrregcount 42" python setup.py install
cd ../scripts
bash quick.sh > dummy; grep 'Latency of Embedding Bag kernel' dummy |  cut -d' ' -f6 | cut -d'm' -f1 > log; python parser_Kernel_ET.py ;
```

### To evaluate L2P + OptMT inference on GPU
```
cd ../pytorch
cp ../opt_designs/l2_pinning/EmbeddingBag_timed.cu aten/src/ATen/native/cuda/EmbeddingBag.cu
export TORCH_CUDA_ARCH_LIST="8.0"
REL_WITH_DEB_INFO=1 USE_NATIVE_ARCH=1 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" USE_CUDA=1 USE_CUDNN=1 USE_NUMPY=1 CMAKE_CUDA_FLAGS="-maxrregcount 42" python setup.py install
cd ../scripts
bash l2_quick.sh > dummy; grep 'Latency of Embedding Bag kernel' dummy |  cut -d' ' -f6 | cut -d'm' -f1 > log; python parser_Kernel_ET.py ;
```

### To evaluate RPF + L2P + OptMT inference on GPU
```
cd ../pytorch
cp ../opt_designs/combined/EmbeddingBag.cu aten/src/ATen/native/cuda/EmbeddingBag.cu
export TORCH_CUDA_ARCH_LIST="8.0"
REL_WITH_DEB_INFO=1 USE_NATIVE_ARCH=1 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" USE_CUDA=1 USE_CUDNN=1 USE_NUMPY=1 CMAKE_CUDA_FLAGS="-maxrregcount 42" python setup.py install
cd ../scripts
bash l2_quick.sh > dummy; grep 'Latency of Embedding Bag kernel' dummy |  cut -d' ' -f6 | cut -d'm' -f1 > log; python parser_Kernel_ET.py ;
```

### Recompile and clean
```
## To clean the build
python setup.py clean
## To recompile
export TORCH_CUDA_ARCH_LIST="8.0"
REL_WITH_DEB_INFO=1 USE_NATIVE_ARCH=1 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" USE_CUDA=1 USE_CUDNN=1 USE_NUMPY=1 python setup.py install
```



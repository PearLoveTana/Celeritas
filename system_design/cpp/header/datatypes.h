#ifndef CELERITAS_DATATYPES_H
#define CELERITAS_DATATYPES_H

#include <exception>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

using std::string;
using std::map;
using std::shared_ptr;
using std::unique_ptr;

/** Program Constants */

#define MAX_READ_SIZE 1E8

/** Deployment configs */

class DummyCuda {
  public:
    DummyCuda(int val) {
        (void) val;
    }

    void start() {};

    void record() {};

    void synchronize() {};

    int elapsed_time(DummyCuda) {
        return 0;
    }
};

#ifdef CELERITAS_CUDA
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAStream.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <ATen/cuda/Exceptions.h>
    #include <c10/util/Exception.h>
    #include <ATen/cuda/CUDAEvent.h>
    typedef at::cuda::CUDAEvent CudaEvent;
#else
typedef DummyCuda CudaEvent;
#endif


#ifndef IO_FLAGS
    #define IO_FLAGS 0
#endif

typedef torch::Tensor EdgeList;

/** Tensor of feature vectors. Shape: (n, FEATURE_SIZE) */
typedef torch::Tensor Features;

/** Single feature vector. Shape (FEATURE_SIZE) */
typedef torch::Tensor Feature;

/** Tensor of integer labels. Shape (n) */
typedef torch::Tensor Labels;

/** Tensor of embedding vectors. Shape: (n, EMBEDDING_SIZE) */
typedef torch::Tensor Embeddings;

/** Single embedding vector. Shape (EMBEDDING_SIZE) */
typedef torch::Tensor Embedding;

/** Tensor of relation vectors. Shape: (n, EMBEDDING_SIZE) */
typedef torch::Tensor Relations;

/** Single relation vector. Shape (EMBEDDING_SIZE) */
typedef torch::Tensor Relation;

/** 1D Tensor of indices. Shape (n) */
typedef torch::Tensor Indices;

/** Tensor of gradients. Shape: (n, EMBEDDING_SIZE) */
typedef torch::Tensor Gradients;

/** Tensor containing optimizer state for a selection of parameters. Shape: (n, FEATURE_SIZE) */
typedef torch::Tensor OptimizerState;

typedef std::chrono::time_point<std::chrono::steady_clock> Timestamp;

#endif 

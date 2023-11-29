#ifndef CELERITAS_BASE_GNN_LAYER_H
#define CELERITAS_BASE_GNN_LAYER_H

#include "config.h"
#include "datatypes.h"
#include "initizlization.h"
#include "graph.h"

class GNNLayer {
public:
    int input_dim_;
    int output_dim_;

    virtual ~GNNLayer() {};

    virtual Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train) {return torch::Tensor();};

};

#endif

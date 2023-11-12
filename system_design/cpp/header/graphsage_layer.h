#ifndef CELERITAS_GRAPHSAGE_LAYER_H
#define CELERITAS_GRAPHSAGE_LAYER_H

#include "base_gnn_layer.h"

class GraphSageLayer : public GNNLayer, public torch::nn::Cloneable<GraphSageLayer> {
public:
    shared_ptr<GNNLayerConfig> layer_config_;
    shared_ptr<GraphSageLayerOptions> options_;
    bool use_incoming_;
    bool use_outgoing_;
    torch::Tensor w1_;
    torch::Tensor w2_;
    torch::Tensor bias_;
    torch::Device device_;

    GraphSageLayer(shared_ptr<GNNLayerConfig> layer_config, bool use_incoming, bool use_outgoing, torch::Device device);

    void reset() override;

    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train = true) override;

};

#endif

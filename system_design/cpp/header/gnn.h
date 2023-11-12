#ifndef CELERITAS_GNN_H
#define CELERITAS_GNN_H

#include "gnn_layer.h"
#include "config.h"

class GeneralGNN : public torch::nn::Cloneable<GeneralGNN> {
public:
    shared_ptr<EncoderConfig> encoder_config_;
    int num_relations_;
    torch::Device device_;

    std::vector<std::shared_ptr<GNNLayer>> layers_;

    GeneralGNN(shared_ptr<EncoderConfig> encoder_config, torch::Device device, int num_relations = 1);

    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train = true);

    void reset() override;

};


#endif

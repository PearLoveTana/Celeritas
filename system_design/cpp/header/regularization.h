//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef CELERITAS_REGULARIZATION_H
#define CELERITAS_REGULARIZATION_H

#include "datatypes.h"

class Regularizer {
  public:
    virtual ~Regularizer() {};

    virtual torch::Tensor operator()(Embeddings src_nodes_embs, Embedding dst_node_embs) = 0;
};

class NormRegularizer : public Regularizer {
  private:
    int norm_;
    float coefficient_;
  public:
    NormRegularizer(int norm, float coefficient);

    torch::Tensor operator()(Embeddings src_nodes_embs, Embedding dst_node_embs) override;
};

#endif 

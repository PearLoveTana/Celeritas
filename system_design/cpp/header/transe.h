#ifndef CELERITAS_TRANSE_H
#define CELERITAS_TRANSE_H

#include "decoder.h"

class TransE : public LinkPredictionDecoder, public torch::nn::Cloneable<TransE> {
public:
    TransE(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations=true);

    void reset() override;
};

#endif

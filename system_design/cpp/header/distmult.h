#ifndef CELERITAS_DISTMULT_H
#define CELERITAS_DISTMULT_H

#include "decoder.h"

class DistMult : public LinkPredictionDecoder, public torch::nn::Cloneable<DistMult> {
public:
    DistMult(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations=true);

    void reset() override;
};

#endif

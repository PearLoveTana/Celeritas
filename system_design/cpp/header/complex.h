#ifndef CELERITAS_COMPLEX_H
#define CELERITAS_COMPLEX_H

#include "decoder.h"

class ComplEx : public LinkPredictionDecoder, public torch::nn::Cloneable<ComplEx> {
public:
    ComplEx(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations=true);

    void reset() override;
};

#endif

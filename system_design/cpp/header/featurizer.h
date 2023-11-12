#ifndef CELERITAS_FEATURIZER_H
#define CELERITAS_FEATURIZER_H

#include "datatypes.h"

class Featurizer : public torch::nn::Module {
  public:
    virtual ~Featurizer() {};
    virtual Embeddings operator()(Features node_features, Embeddings node_embeddings) = 0;
};

class CatFeaturizer : public Featurizer {
  public:
    CatFeaturizer(int norm, float coefficient);

    Embeddings operator()(Features node_features, Embeddings node_embeddings) override;
};


#endif

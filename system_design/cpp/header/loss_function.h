#ifndef CELERITAS_SRC_CPP_INCLUDE_LOSS_H_
#define CELERITAS_SRC_CPP_INCLUDE_LOSS_H_

#include "config.h"
#include "datatypes.h"


class LossFunction {
  public:
    virtual ~LossFunction() {};
    virtual torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) = 0;
};

class SoftMax : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    SoftMax(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class RankingLoss : public LossFunction {
  private:
    float margin_;
    LossReduction reduction_type_;
  public:
    RankingLoss(shared_ptr<RankingLossOptions> options) {
        margin_ = options->margin;
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEAfterSigmoidLoss: public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    BCEAfterSigmoidLoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEWithLogitsLoss : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    BCEWithLogitsLoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class MSELoss : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    MSELoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class SoftPlusLoss : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    SoftPlusLoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

shared_ptr<LossFunction> getLossFunction(shared_ptr<LossConfig> config);

#endif 

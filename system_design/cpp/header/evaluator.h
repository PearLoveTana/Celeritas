#ifndef CELERITAS_EVALUATOR_H
#define CELERITAS_EVALUATOR_H

#include <iostream>

#include "dataloader.h"
#include "pipeline.h"

/**
  The evaluator runs the evaluation process using the given model and dataset.
*/
class Evaluator {
  public:
    DataLoader *dataloader_;

    virtual ~Evaluator() { };
    virtual void evaluate(bool validation) = 0;
};

class PipelineEvaluator : public Evaluator {
    Pipeline *pipeline_;
  public:
    PipelineEvaluator(DataLoader *sampler, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config);

    void evaluate(bool validation) override;
};

class SynchronousEvaluator : public Evaluator {
    shared_ptr<Model> model_;
  public:
    SynchronousEvaluator(DataLoader *sampler, shared_ptr<Model> model);

    void evaluate(bool validation) override;
};

#endif 


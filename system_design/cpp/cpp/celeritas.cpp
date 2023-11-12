#include "celeritas.h"

#include "config.h"
#include "evaluator.h"
#include "io_manager.h"
#include "logger.h"
#include "model.h"
#include "train.h"
#include "utils.h"

void celeritas(int argc, char *argv[]) {

    bool train = true;
    string command_path = string(argv[0]);
    string config_path = string(argv[1]);
    string command_name = command_path.substr(command_path.find_last_of("/\\") + 1);
    if (strcmp(command_name.c_str(), "celeritas_eval") == 0) {
        train = false;
    }

    shared_ptr<CeleritasConfig> celeritas_config = initConfig(config_path);

    torch::manual_seed(celeritas_config->model->random_seed);
    srand(celeritas_config->model->random_seed);

    Timer initialization_timer = Timer(false);
    initialization_timer.start();
    SPDLOG_INFO("Start initialization");

    std::vector<torch::Device> devices;

    if (celeritas_config->storage->device_type == torch::kCUDA) {
        for (int i = 0; i < celeritas_config->storage->device_ids.size(); i++) {
            devices.emplace_back(torch::Device(torch::kCUDA, celeritas_config->storage->device_ids[i]));
        }

        if (devices.empty()) {
            devices.emplace_back(torch::Device(torch::kCUDA, 0));
        }
    } else {
        devices.emplace_back(torch::kCPU);
    }

    std::shared_ptr<Model> model = initializeModel(celeritas_config->model,
                                                   devices,
                                                   celeritas_config->storage->dataset->num_relations);
    model->train_ = train;

    if (celeritas_config->evaluation->negative_sampling != nullptr) {
        model->filtered_eval_ = celeritas_config->evaluation->negative_sampling->filtered;
    } else {
        model->filtered_eval_ = false;
    }

    GraphModelStorage *graph_model_storage = initializeStorage(model, celeritas_config->storage);

    DataLoader *dataloader = new DataLoader(graph_model_storage,
                                            celeritas_config->training,
                                            celeritas_config->evaluation,
                                            celeritas_config->model->encoder);

    initialization_timer.stop();
    int64_t initialization_time = initialization_timer.getDuration();

    SPDLOG_INFO("Initialization Complete: {}s", (double) initialization_time / 1000);

    Trainer *trainer;
    Evaluator *evaluator;

    if (train) {
        if (celeritas_config->training->pipeline->sync) {
            if (celeritas_config->storage->device_ids.size() > 1) {
                trainer = new SynchronousMultiGPUTrainer(dataloader, model, celeritas_config->training->logs_per_epoch);
            } else {
                trainer = new SynchronousTrainer(dataloader, model, celeritas_config->training->logs_per_epoch);
            }
        } else {
            trainer = new PipelineTrainer(dataloader,
                                          model,
                                          celeritas_config->training->pipeline,
                                          celeritas_config->training->logs_per_epoch);
        }

        if (celeritas_config->evaluation->pipeline->sync) {
            evaluator = new SynchronousEvaluator(dataloader, model);
        } else {
            evaluator = new PipelineEvaluator(dataloader,
                                              model,
                                              celeritas_config->evaluation->pipeline);
        }

        for (int epoch = 0; epoch < celeritas_config->training->num_epochs; epoch++) {
            if ((epoch + 1) % celeritas_config->evaluation->epochs_per_eval != 0) {
                trainer->train(1);
            } else {
                trainer->train(1);
                evaluator->evaluate(true);
                evaluator->evaluate(false);
            }
        }
    } else {
        if (celeritas_config->evaluation->pipeline->sync) {
            evaluator = new SynchronousEvaluator(dataloader, model);
        } else {
            evaluator = new PipelineEvaluator(dataloader,
                                              model,
                                              celeritas_config->evaluation->pipeline);
        }
        evaluator->evaluate(false);
    }

    model->save(celeritas_config->storage->dataset->base_directory);

    // garbage collect
    delete graph_model_storage;
    delete trainer;
    delete evaluator;
    delete dataloader;
}

int main(int argc, char *argv[]) {
    celeritas(argc, argv);
}
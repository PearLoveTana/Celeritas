#ifndef CELERITAS_LAYER_UTILS_H
#define CELERITAS_LAYER_UTILS_H

#include "datatypes.h"

torch::Tensor segment_ids_from_offsets(torch::Tensor offsets, int64_t input_size);

torch::Tensor segmented_sum(torch::Tensor tensor, torch::Tensor segment_ids, int64_t num_segments);

torch::Tensor segmented_sum_with_offsets(torch::Tensor tensor, torch::Tensor offsets);

//torch::Tensor segmented_max(torch::Tensor tensor, torch::Tensor segment_ids, int64_t num_segments);

torch::Tensor segmented_max_with_offsets(torch::Tensor tensor, torch::Tensor offsets);

std::tuple<torch::Tensor, torch::Tensor> attention_softmax(torch::Tensor neighbor_attention,
                                                           torch::Tensor self_attention,
                                                           torch::Tensor segment_offsets,
                                                           torch::Tensor segment_ids,
                                                           torch::Tensor num_nbrs);

#endif 

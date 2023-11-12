#include "comparators.h"

std::tuple<torch::Tensor, torch::Tensor> L2Compare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int) ceil((float) num_pos / num_chunks);

    Embeddings adjusted_src = src;
    Embeddings adjusted_dst = dst;

    // pad embedding tensor if the number of elements is not divisible by the number of chunks
    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        adjusted_src = torch::nn::functional::pad(adjusted_src, options);
        adjusted_dst = torch::nn::functional::pad(adjusted_dst, options);
    }

    adjusted_src = adjusted_src.unsqueeze(1);
    adjusted_dst = adjusted_dst.unsqueeze(1);
    torch::Tensor pos_scores = torch::cdist(adjusted_src, adjusted_dst).flatten(0, 2);

    adjusted_src = adjusted_src.view({num_chunks, num_per_chunk, src.size(1)});
    // compute batched distance between source nodes and negative nodes
    torch::Tensor neg_scores = torch::cdist(adjusted_src, negs).flatten(0, 1);

    return std::make_tuple(std::move(pos_scores), std::move(neg_scores));
}


std::tuple<torch::Tensor, torch::Tensor> CosineCompare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int64_t) ceil((float) num_pos / num_chunks);

    torch::Tensor src_norm = src.norm(2, -1);
    torch::Tensor dst_norm = dst.norm(2, -1);
    torch::Tensor neg_norm = negs.norm(2, -1);

    Embeddings normalized_src = src * src_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);
    Embeddings normalized_dst = dst * dst_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);
    Embeddings normalized_neg = negs * neg_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        normalized_src = torch::nn::functional::pad(normalized_src, options);
        normalized_dst = torch::nn::functional::pad(normalized_dst, options);
    }

    torch::Tensor pos_scores = (normalized_src * normalized_dst).sum(-1);
    normalized_src = normalized_src.view({num_chunks, num_per_chunk, normalized_src.size(1)});
    torch::Tensor neg_scores = normalized_src.bmm(normalized_neg.transpose(-1, -2)).flatten(0, 1);

    return std::make_tuple(std::move(pos_scores), std::move(neg_scores));
}

std::tuple<torch::Tensor, torch::Tensor> DotCompare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int) ceil((float) num_pos / num_chunks);

    Embeddings adjusted_src = src;
    Embeddings adjusted_dst = dst;

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        adjusted_src = torch::nn::functional::pad(adjusted_src, options);
        adjusted_dst = torch::nn::functional::pad(adjusted_dst, options);
    }

    torch::Tensor pos_scores = (adjusted_src * adjusted_dst).sum(-1);
    adjusted_src = adjusted_src.view({num_chunks, num_per_chunk, src.size(1)});

    torch::Tensor neg_scores = adjusted_src.bmm(negs.transpose(-1, -2)).flatten(0, 1);

    return std::make_tuple(std::move(pos_scores), std::move(neg_scores));
}
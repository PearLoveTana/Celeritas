#include <torch/extension.h>

#include "graph.h"

namespace py = pybind11;

void init_graph(py::module &m) {

    py::class_<CeleritasGraph>(m, "CeleritasGraph")
        .def_readwrite("src_sorted_edges", &CeleritasGraph::src_sorted_edges_)
        .def_readwrite("dst_sorted_edges", &CeleritasGraph::dst_sorted_edges_)
        .def_readwrite("active_in_memory_subgraph", &CeleritasGraph::active_in_memory_subgraph_)
        .def_readwrite("num_nodes_in_memory", &CeleritasGraph::num_nodes_in_memory_)
        .def_readwrite("node_ids", &CeleritasGraph::node_ids_)
        .def_readwrite("out_sorted_uniques", &CeleritasGraph::out_sorted_uniques_)
        .def_readwrite("out_offsets", &CeleritasGraph::out_offsets_)
        .def_readwrite("out_num_neighbors", &CeleritasGraph::out_num_neighbors_)
        .def_readwrite("in_sorted_uniques", &CeleritasGraph::in_sorted_uniques_)
        .def_readwrite("in_offsets", &CeleritasGraph::in_offsets_)
        .def_readwrite("in_num_neighbors", &CeleritasGraph::in_num_neighbors_)
        .def_readwrite("max_out_num_neighbors_", &CeleritasGraph::max_out_num_neighbors_)
        .def_readwrite("max_in_num_neighbors_", &CeleritasGraph::max_in_num_neighbors_)
        .def(py::init<>())
        .def(py::init<EdgeList, EdgeList, int64_t, int>(), py::arg("src_sorted_edges"), py::arg("dst_sorted_edges"), py::arg("num_nodes_in_memory"), py::arg("num_hash_maps"))
        .def("getEdges", &CeleritasGraph::getEdges, py::arg("incoming") = true)
        .def("getRelationIDs", &CeleritasGraph::getRelationIDs, py::arg("incoming") = true)
        .def("getNeighborOffsets", &CeleritasGraph::getNeighborOffsets, py::arg("incoming") = true)
        .def("getNumNeighbors", &CeleritasGraph::getNumNeighbors, py::arg("incoming") = true)
        .def("getNeighborsForNodeIds", &CeleritasGraph::getNeighborsForNodeIds, py::arg("node_ids"), py::arg("incoming"), py::arg("neighbor_sampling_layer"), py::arg("max_neighbors_size"), py::arg("rate"))
        .def("clear", &CeleritasGraph::clear)
        .def("to", &CeleritasGraph::to, py::arg("device"));

    py::class_<GNNGraph, CeleritasGraph>(m, "GNNGraph")
        .def_readwrite("hop_offsets", &GNNGraph::hop_offsets_)
        .def_readwrite("in_neighbors_mapping", &GNNGraph::in_neighbors_mapping_)
        .def_readwrite("out_neighbors_mapping", &GNNGraph::out_neighbors_mapping_)
        .def_readwrite("in_neighbors_vec", &GNNGraph::in_neighbors_vec_)
        .def_readwrite("out_neighbors_vec", &GNNGraph::out_neighbors_vec_)
        .def_readwrite("node_properties", &GNNGraph::node_properties_)
        .def_readwrite("num_nodes_in_memory", &GNNGraph::num_nodes_in_memory_)
        .def(py::init<>())
        .def(py::init<Indices, Indices, Indices, std::vector<torch::Tensor>, Indices, Indices, std::vector<torch::Tensor>, Indices, int>(), 
            py::arg("hop_offsets"), 
            py::arg("node_ids"), 
            py::arg("in_offsets"), 
            py::arg("in_neighbors_vec"), 
            py::arg("in_neighbors_mapping"), 
            py::arg("out_offsets"), 
            py::arg("out_neighbors_vec"), 
            py::arg("out_neighbors_mapping"), 
            py::arg("num_nodes_in_memory"))
        .def("prepareForNextLayer", &GNNGraph::prepareForNextLayer)
        .def("getNeighborIDs", &GNNGraph::getNeighborIDs, py::arg("incoming") = true, py::arg("global") = false)
        .def("getLayerOffset", &GNNGraph::getLayerOffset)
        .def("performMap", &GNNGraph::performMap)
        .def("setNodeProperties", &GNNGraph::setNodeProperties, py::arg("node_properties"))
        .def("clear", &GNNGraph::clear)
        .def("to", [](GNNGraph &graph, torch::Device device) {
            graph.to(device, nullptr, nullptr);
        }, py::arg("device"));
}
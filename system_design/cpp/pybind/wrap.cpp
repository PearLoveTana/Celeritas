#define PYBIND11_COMPILER_TYPE ""
#define PYBIND11_STDLIB ""
#define PYBIND11_BUILD_ABI ""

#include "torch/extension.h"

namespace py = pybind11;

void init_config(py::module &);
void init_options(py::module &);

void init_activation(py::module &);
void init_batch(py::module &);
void init_dataloader(py::module &);
void init_datatypes(py::module &);
void init_evaluator(py::module &);
void init_graph_samplers(py::module &);
void init_graph_storage(py::module &);
void init_graph(py::module &);
void init_initialization(py::module &);
void init_io(py::module &);
void init_loss(py::module &);
void init_celeritas(py::module &);
void init_model(py::module &);
void init_regularizer(py::module &);
void init_reporting(py::module &);
void init_trainer(py::module &);

PYBIND11_MODULE(_pyceleritas, m) {

	m.doc() = "pybind11 celeritas plugin";

    // configuration
    init_config(m);
    init_options(m);
    init_activation(m);
    init_batch(m);
    init_dataloader(m);
    init_datatypes(m);
    init_evaluator(m);
    init_graph_samplers(m);
    init_graph_storage(m);
    init_graph(m);
    init_initialization(m);
    init_io(m);
    init_loss(m);
    init_marius(m);
    init_model(m);
    init_regularizer(m);
    init_reporting(m);
    init_trainer(m);
}
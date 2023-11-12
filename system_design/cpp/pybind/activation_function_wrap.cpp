#include <torch/extension.h>

#include "activation_function.h"

namespace py = pybind11;

void init_activation(py::module &m) {

    m.def("apply_activation", &apply_activation, py::arg("activation_function"), py::arg("input"));
}
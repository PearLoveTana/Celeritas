//
// Created by Jason Mohoney on 4/9/21.
//

#include <torch/extension.h>

#include "celeritas.h"

namespace py = pybind11;

void init_celeritas(py::module &m) {
    m.def("celeritas_train", [](int argc, std::vector<std::string> argv) {

            argv[0] = "celeritas_train";
            std::vector<char *> c_strs;
            c_strs.reserve(argv.size());
            for (auto &s : argv) c_strs.push_back(const_cast<char *>(s.c_str()));

            celeritas(argc, c_strs.data());

        }, py::arg("argc"), py::arg("argv"), py::return_value_policy::reference);

    m.def("celeritas_eval", [](int argc, std::vector<std::string> argv) {

        argv[0] = "celeritas_eval";
        std::vector<char *> c_strs;
        c_strs.reserve(argv.size());
        for (auto &s : argv) c_strs.push_back(const_cast<char *>(s.c_str()));

        celeritas(argc, c_strs.data());

    }, py::arg("argc"), py::arg("argv"), py::return_value_policy::reference);
}
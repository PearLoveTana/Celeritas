#ifndef CELERITAS_ACTIVATION_FUNCTION_H
#define CELERITAS_ACTIVATION_FUNCTION_H

#include "config.h"
#include "datatypes.h"

torch::Tensor apply_activation(ActivationFunction activation_function, torch::Tensor input);

#endif

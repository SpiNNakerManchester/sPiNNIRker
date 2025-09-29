/*
 Copyright (c) 2025 The University of Manchester
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
//! \file affine_common.h
//! \brief Affine NIR common implementation

#include <stdint.h>
#include <spin1_api.h>
#include <debug.h>
#include "../component.h"
#include "matrix_common.h"

//! The configuration passed to the component
typedef struct {
    //! The height of the input / output
    uint32_t io_height;

    // The common matrix parameters
    matrix_config_t matrix;
    // The common bias parameters follows (disallowed in C)
    // matrix_config_t bias;
} affine_common_config_t;

typedef struct {
    //! The height of the input / output
    union {
        uint32_t input_height;
        uint32_t output_height;
    };
    //! The common matrix data
    matrix_data_t weights_data;
    //! The bias data
    matrix_data_t bias_data;
} affine_common_data_t;

static affine_common_data_t *affine_common_init(uint32_t index,
        affine_common_config_t *config, affine_common_data_t *data) {
    matrix_init(index, &config->matrix, &data->weights_data);
    matrix_config_t *bias_config = (matrix_config_t *)
            (&config->matrix.data[config->matrix.width * config->matrix.height]);
    matrix_init(index, bias_config, &data->bias_data);
    data->input_height = config->io_height;
    return data;
}

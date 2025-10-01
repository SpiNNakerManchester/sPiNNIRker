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
//! \file linear_common.h
//! \brief Linear NIR common implementation

#include "matrix_common.h"

//! The configuration passed to the component
typedef struct {
    //! The height of the input / output
    uint32_t io_height;

    // The common matrix parameters
    matrix_config_t matrix;
} linear_common_config_t;

typedef struct {
    //! The height of the input / output
    union {
        uint32_t input_height;
        uint32_t output_height;
    };
    //! The common matrix data
    matrix_data_t weights_data;
} linear_common_data_t;

static linear_common_data_t *linear_common_init(uint32_t index,
        linear_common_config_t *config, linear_common_data_t *data) {
    matrix_init(index, &config->matrix, &data->weights_data, sizeof(int32_t));
    data->input_height = config->io_height;
    return data;
}

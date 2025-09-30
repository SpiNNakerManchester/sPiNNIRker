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
//! \file affine_matrix.c
//! \brief Affine NIR component implementation (matrix inputs)
//! This is a 2D matrix multiplication of inputs by weights with an additional
//! bias

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "affine_matrix.h"
#include "affine_common.h"
#include "matrix_matrix_common.h"
#include "matrix_common_multiply.h"
#include "matrix_clear.h"

static void *affine_matrix_init(uint32_t index, void *params) {
    affine_common_data_t *data = spin1_malloc(sizeof(affine_common_data_t));
    if (!data) {
        log_error("Failed to allocate affine matrix data structure");
        return (void *)0;
    }
    return affine_common_init(index, params, data);
}

static void affine_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    affine_common_data_t *affine_data = data;

    matrix_clear_outputs(output, affine_data->output_height *
            affine_data->weights_data.width);

    matrix_loop_t loop = matrix_loop_start(&affine_data->weights_data);

    int32_t value;
    uint32_t row;
    uint32_t col;
    while (matrix_loop_is_next(&loop, &value, &row, &col)) {
        matrix_matrix_multiply(loop.data, row, col, loop.current_data, input,
            n_inputs, output.data);
    }

    // Now run a loop over the biases and add them in to the output
    int32_t *out_data = output.data;
    matrix_loop_t bias_loop = matrix_loop_start(&affine_data->bias_data);
    while (matrix_loop_is_next(&bias_loop, &value, &row, &col)) {
        uint32_t i_off = row * affine_data->bias_data.width;
        out_data[i_off + col] += value;
    }
}

const component_t affine_matrix = {
    .init = affine_matrix_init,
    .func = affine_matrix_exec,
    .dma_complete = matrix_common_dma_complete
};

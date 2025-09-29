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
//! \file affine_spikes.c
//! \brief Affine NIR component implementation (spike inputs)
//! This is a 2D matrix multiplication of inputs by weights, but using
//! spikes for improved efficiency.

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "affine_spikes.h"
#include "affine_common.h"
#include "matrix_common_multiply.h"
#include "matrix_matrix_common.h"
#include "matrix_spikes_common.h"
#include "matrix_clear.h"

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    affine_common_config_t base_config;
} affine_spikes_config_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    affine_common_data_t base_config;
} affine_spikes_data_t;

static void *affine_spikes_init(uint32_t index, void *params) {
    affine_spikes_data_t *data = spin1_malloc(sizeof(affine_spikes_data_t));
    if (!data) {
        log_error("Failed to allocate affine matrix data structure");
        return (void *)0;
    }
    affine_spikes_config_t *spikes_config = params;
    affine_common_init(index, &spikes_config->base_config, &data->base_config);
    data->input_width_inv = spikes_config->input_width_inv;
    return data;
}

static void affine_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    affine_spikes_data_t *spikes_data = data;
    affine_common_data_t *linear_data = &spikes_data->base_config;
    matrix_data_t *matrix_data = &linear_data->weights_data;

    // Reset out data
    matrix_clear_outputs(output, matrix_data->width);

    // For simplicity, the input width = the matrix height
    uint32_t input_width = matrix_data->height;

    // Start a loop - if there are no spikes, we are done
    matrix_spikes_loop_data_t loop = matrix_spikes_loop_start(input, n_inputs,
            input_width, spikes_data->input_width_inv, 1, matrix_data);

    int32_t value;
    uint32_t row;
    uint32_t col;
    while (matrix_spikes_loop_is_next(&loop, &value, &row, &col)) {
        matrix_matrix_multiply(loop.matrix_data, row, col, loop.current_data,
            input, n_inputs, output.data);
    }

    // Now run a loop over the biases and add them in to the output
    int32_t *out_data = output.data;
    matrix_loop_t bias_loop = matrix_loop_start(
        &spikes_data->base_config.bias_data);
    while (matrix_loop_is_next(&bias_loop, &value, &row, &col)) {
        uint32_t i_off = row * spikes_data->base_config.bias_data.width;
        out_data[i_off + col] += value;
    }
}

const component_t affine_spikes = {
    .init = affine_spikes_init,
    .func = affine_spikes_exec,
    .dma_complete = matrix_common_dma_complete
};

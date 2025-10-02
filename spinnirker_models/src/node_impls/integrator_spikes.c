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
//! \file integrator_spikes.c
//! \brief integrator NIR component implementation (spike inputs)
//! This is a 2D matrix multiplication of inputs by weights, but using
//! spikes for improved efficiency.

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "integrator_spikes.h"
#include "matrix_spikes_common.h"
#include "matrix_clear.h"

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    matrix_config_t integrator_matrix;
} integrator_spikes_config_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    matrix_data_t integrator_matrix;
} integrator_spikes_data_t;

static void *integrator_spikes_init(uint32_t index, void *params) {
    integrator_spikes_data_t *data = spin1_malloc(sizeof(integrator_spikes_data_t));
    if (!data) {
        log_error("Failed to allocate integrator matrix data structure");
        return (void *)0;
    }
    integrator_spikes_config_t *spikes_config = params;
    matrix_init(index, &spikes_config->integrator_matrix,
        &data->integrator_matrix, sizeof(int32_t));
    data->input_width_inv = spikes_config->input_width_inv;
    return data;
}

static void integrator_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    integrator_spikes_data_t *spikes_data = data;
    matrix_data_t *integrator_data = &spikes_data->integrator_matrix;
    int32_t *out_data = output.data;

    // For each spike, add the corresponding row of the matrix to the output,
    // since this is the input value of 1 multiplied by the resistance
    matrix_spikes_loop_data_t loop = matrix_spikes_loop_start(input, n_inputs,
            integrator_data->width, spikes_data->input_width_inv, 0,
            integrator_data);
    uint32_t row;
    uint32_t col;
    while (matrix_spikes_loop_is_next(&loop, &row, &col)) {
        int32_t *row_data = loop.current_data;
        uint32_t i_off = row * integrator_data->width;
        out_data[i_off + col] += row_data[col];
    }
}

const component_t integrator_spikes = {
    .init = integrator_spikes_init,
    .func = integrator_spikes_exec,
    .dma_complete = matrix_common_dma_complete
};

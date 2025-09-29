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
//! \file scale_spikes.c
//! \brief scale NIR component implementation (spike inputs)
//! This is a 2D matrix multiplication of inputs by weights, but using
//! spikes for improved efficiency.

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "scale_spikes.h"
#include "matrix_spikes_common.h"
#include "matrix_clear.h"

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    matrix_config_t scale_matrix;
} scale_spikes_config_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    matrix_data_t scale_matrix;
} scale_spikes_data_t;

static void *scale_spikes_init(uint32_t index, void *params) {
    scale_spikes_data_t *data = spin1_malloc(sizeof(scale_spikes_data_t));
    if (!data) {
        log_error("Failed to allocate scale matrix data structure");
        return (void *)0;
    }
    scale_spikes_config_t *spikes_config = params;
    matrix_init(index, &spikes_config->scale_matrix, &data->scale_matrix);
    data->input_width_inv = spikes_config->input_width_inv;
    return data;
}

static void scale_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    scale_spikes_data_t *spikes_data = data;
    matrix_data_t *scale_data = &spikes_data->scale_matrix;
    int32_t *out_data = output.data;

    matrix_clear_outputs(output, scale_data->width * scale_data->height);

    // Start a loop - if there are no spikes, we are done
    matrix_spikes_loop_data_t loop = matrix_spikes_loop_start(input, n_inputs,
            scale_data->width, spikes_data->input_width_inv, 0, scale_data);

    uint32_t row;
    uint32_t col;
    int32_t value;
    while (matrix_spikes_loop_is_next(&loop, &value, &row, &col)) {
        // Go through and sum the inputs
        int32_t acc = 0;
        uint32_t i_off = row * scale_data->width;
        for (uint32_t k = 0; k < n_inputs; k++) {
            int32_t *in_data = input[k].data;
            acc += in_data[i_off + col];
        }
        out_data[i_off + col] = __stdfix_smul_k(acc, value);
    }
}

const component_t scale_spikes = {
    .init = scale_spikes_init,
    .func = scale_spikes_exec,
    .dma_complete = matrix_common_dma_complete
};

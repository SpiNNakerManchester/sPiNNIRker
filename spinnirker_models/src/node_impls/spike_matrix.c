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
//! \file spike_matrix.c
//! \brief spike NIR component implementation (matrix inputs)

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "spike_matrix.h"
#include "matrix_matrix_common.h"

typedef struct {
    uint32_t key;
    matrix_config_t matrix_data;
} spike_matrix_config_t;

typedef struct {
    uint32_t key;
    matrix_data_t matrix_data;
} spike_matrix_data_t;

static void *spike_matrix_init(uint32_t index, void *params) {
    spike_matrix_data_t *data = spin1_malloc(sizeof(spike_matrix_data_t));
    if (!data) {
        log_error("Failed to allocate spike matrix data structure");
        return (void *)0;
    }
    spike_matrix_config_t *config = params;
    data->key = config->key;
    return matrix_init(index, &config->matrix_data, &data->matrix_data,
            sizeof(int32_t));
}

static void spike_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    spike_matrix_data_t *spike_data = data;
    matrix_data_t *thresh_data = &spike_data->matrix_data;

    matrix_loop_t loop = matrix_loop_start(thresh_data);
    int32_t *out_data = output.data;
    uint32_t row;
    uint32_t col;
    while (matrix_loop_is_next(&loop, &row, &col)) {
        int32_t *row_data = loop.current_data;
        // Go through and sum the inputs
        int32_t acc = 0;
        uint32_t i_off = row * thresh_data->width;
        for (uint32_t k = 0; k < n_inputs; k++) {
            int32_t *in_data = input[k].data;
            acc += in_data[i_off + col];
        }
        uint32_t spike = acc >= row_data[col];
        out_data[i_off + col] = spike;
        if (spike) {
            spin1_send_mc_packet(spike_data->key + i_off + col, 0, 0);
        }
    }
}

const component_t spike_matrix = {
        .init = spike_matrix_init,
        .func = spike_matrix_exec,
        .dma_complete = matrix_common_dma_complete
};

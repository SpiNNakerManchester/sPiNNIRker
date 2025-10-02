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
//! \file spike_spikes.c
//! \brief spike NIR component implementation (spike inputs)

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "spike_spikes.h"
#include "matrix_spikes_common.h"
#include "matrix_clear.h"

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    //! The spike key to send
    uint32_t key;

    matrix_config_t spike_matrix;
} spike_spikes_config_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    //! The spike key to send
    uint32_t key;

    matrix_data_t spike_matrix;
} spike_spikes_data_t;

static void *spike_spikes_init(uint32_t index, void *params) {
    spike_spikes_data_t *data = spin1_malloc(sizeof(spike_spikes_data_t));
    if (!data) {
        log_error("Failed to allocate spike matrix data structure");
        return (void *)0;
    }
    spike_spikes_config_t *spikes_config = params;
    matrix_init(index, &spikes_config->spike_matrix, &data->spike_matrix,
            sizeof(int32_t));
    data->input_width_inv = spikes_config->input_width_inv;
    data->key = spikes_config->key;
    return data;
}

static void spike_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    spike_spikes_data_t *spike_data = data;
    matrix_data_t *thresh_data = &spike_data->spike_matrix;
    int32_t *out_data = output.data;

    // We reset the outputs here since where there are no spikes the output is
    // definitely 0
    matrix_clear_outputs(output, thresh_data->width * thresh_data->height);

    // Go through each spike and see if this 1 value makes the output go over
    // the thresold
    matrix_spikes_loop_data_t loop = matrix_spikes_loop_start(input, n_inputs,
            thresh_data->width, spike_data->input_width_inv, 0, thresh_data);
    uint32_t row;
    uint32_t col;
    while (matrix_spikes_loop_is_next(&loop, &row, &col)) {
        int32_t *row_data = loop.current_data;
        uint32_t i_off = row * thresh_data->width;
        uint32_t spike = 1 >= row_data[col];
        out_data[i_off + col] = spike;
        if (spike) {
            spin1_send_mc_packet(spike_data->key + i_off + col, 0, 0);
        }
    }
}

const component_t spike_spikes = {
    .init = spike_spikes_init,
    .func = spike_spikes_exec,
    .dma_complete = matrix_common_dma_complete
};

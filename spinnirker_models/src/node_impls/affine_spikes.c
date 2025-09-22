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

static uint32_t is_next_spike(affine_spikes_data_t *affine_data, uint32_t row,
        uint32_t n_inputs, data_t *input, uint32_t *input_index,
        uint32_t *index, uint32_t *i_row) {
    while (*input_index < n_inputs) {
        spike_list_t *spikes = input[*input_index].data;
        while (*index < spikes->n_spikes) {
            uint32_t source_id = spikes->spikes[*index].global_source_id;
            *i_row = div_by_const(source_id, affine_data->input_width_inv);
            uint32_t col = source_id -
                           (*i_row * affine_data->base_config.input_width);
            if (row == col) {
                return 1;
            }
            (*index)++;
        }
        *index = 0;
        (*input_index)++;
    }
    return 0;
}

static uint32_t affine_spikes_find_next_weight_row(
        affine_spikes_data_t *affine_data, uint32_t row, uint32_t n_inputs,
        data_t *input) {
    uint32_t input_index = 0;
    uint32_t index = 0;
    uint32_t i_row = 0;
    while (row < affine_data->base_config.weights_height &&
            !is_next_spike(affine_data, row, n_inputs, input, &input_index,
                    &index, &i_row)) {
        row++;
    }
    return row;
}

static void affine_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    affine_spikes_data_t *spikes_data = data;
    affine_common_data_t *affine_data = &spikes_data->base_config;

    // Convert to right type for output (Accum but only ever added to)
    // and clear
    int32_t *out_data = output.data;
    for (uint32_t i = 0; i < affine_data->output_height; i++) {
        uint32_t i_off = i * affine_data->output_width;
        for (uint32_t j = 0; j < affine_data->output_width; j++) {
            out_data[i_off + j] = 0;
        }
    }

    // Find the first row of weights we need
    uint32_t k = affine_spikes_find_next_weight_row(data, 0, n_inputs, input);
    transfer_weights(affine_data, k);

    // Go through the rows until the end of the weights
    while (k < affine_data->weights_height) {
        // Wait for the weights to be ready
        int32_t *weights = get_weights(affine_data, k);

        // Start the transfer of the next row (will be ignored if last row)
        uint32_t next_k = affine_spikes_find_next_weight_row(data, k + 1,
                n_inputs, input);
        transfer_weights(affine_data, next_k);

        // Go through the spikes and process those for this row of weights
        // i.e. those where the column matches the row
        uint32_t input_index = 0;
        uint32_t index = 0;
        uint32_t i_row = 0;
        while (is_next_spike(spikes_data, k, n_inputs, input, &input_index,
                &index, &i_row)) {
            // Offset of input row i in output
            uint32_t i_off_out = i_row * affine_data->output_width;
            // Add the weights to each output column (as the input here is 1,
            // so this is multiply by 1))
            for (uint32_t j = 0; j < affine_data->weights_width; j++) {
                out_data[i_off_out + j] += weights[j];
            }
        }

        // Now start at the next weight row that is valid (or beyond end)
        k = next_k;
    }

    // Now run a loop over the biases and add them in to the output
    // Request the first row of biases
    transfer_biases(affine_data, 0);
    for (uint32_t j = 0; j < affine_data->output_height; j++) {
        // Wait for the biases to be ready
        int32_t *biases = get_biases(affine_data, j);

        // Start the transfer of the next row (will be ignored if last row)
        transfer_biases(affine_data, j + 1);

        // Add the biases to the output row
        for (uint32_t i = 0; i < affine_data->output_width; i++) {
            uint32_t i_off_out = i * affine_data->output_width;
                out_data[i_off_out + i] += biases[i];
        }
    }
}

const component_t affine_spikes = {
    .init = affine_spikes_init,
    .func = affine_spikes_exec,
    .deinit = affine_common_deinit,
    .dma_complete = affine_common_dma_complete
};

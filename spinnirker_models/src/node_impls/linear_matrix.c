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
//! \file input.c
//! \brief Linear NIR component implementation (matrix inputs)
//! This is a 2D matrix multiplication of inputs by weights.

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "linear_matrix.h"
#include "linear_common.h"

static void *linear_matrix_init(uint32_t index, void *params) {
    linear_common_data_t *data = spin1_malloc(sizeof(linear_common_data_t));
    if (!data) {
        log_error("Failed to allocate linear matrix data structure");
        return (void *)0;
    }
    return linear_common_init(index, params, data);
}

static void linear_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    linear_common_data_t *linear_data = data;

    // Convert to right type for output (Accum but only ever assigned to)
    // and clear
    int32_t *out_data = output.data;
    for (uint32_t i = 0; i < linear_data->output_height; i++) {
        uint32_t i_off = i * linear_data->output_width;
        for (uint32_t j = 0; j < linear_data->output_width; j++) {
            out_data[i_off + j] = 0;
        }
    }

    // Request the first row of weights
    transfer_weights(linear_data, 0);

    // Run the loop over the weights, as those might be in SDRAM.
    // This means we are doing matrix multiplication AxB = C by the rows of B
    // rather than by the rows of C, meaning this will look a little odd...
    for (uint32_t k = 0; k < linear_data->weights_height; k++) {
        // Wait for the weights to be ready
        int32_t *weights = get_weights(linear_data, k);

        // Start the transfer of the next row (will be ignored if last row)
        transfer_weights(linear_data, k + 1);

        // Go through the row of weights
        for (uint32_t j = 0; j < linear_data->weights_width; j++) {
            // Go through column k of each of the input rows
            for (uint32_t i = 0; i < linear_data->input_height; i++) {
                // Offset of row i in input
                uint32_t i_off_in = i * linear_data->input_width;
                // Offset of row i in output
                uint32_t i_off_out = i * linear_data->output_width;

                // Add up each of the inputs
                int32_t sum = 0;
                for (uint32_t idx = 0; idx < n_inputs; idx++) {
                    sum += ((int32_t *)input[idx].data)[i_off_in + k];
                }

                // Add the product of input sum and weight to the output
                out_data[i_off_out + j] = __stdfix_smul_k(sum, weights[j]);
            }
        }
    }
}

const component_t linear_matrix = {
        .init = linear_matrix_init,
        .func = linear_matrix_exec,
        .deinit = linear_common_deinit,
        .dma_complete = linear_common_dma_complete};

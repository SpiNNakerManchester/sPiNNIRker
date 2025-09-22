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
//! \file scale_matrix.c
//! \brief scale NIR component implementation (matrix inputs)
//! This is a 2D matrix multiplication of inputs by weights.

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "scale_matrix.h"
#include "scale_common.h"

static void *scale_matrix_init(uint32_t index, void *params) {
    scale_common_data_t *data = spin1_malloc(sizeof(scale_common_data_t));
    if (!data) {
        log_error("Failed to allocate scale matrix data structure");
        return (void *)0;
    }
    return scale_common_init(index, params, data);
}

static void scale_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    scale_common_data_t *scale_data = data;

    // Get the output data pointer
    int32_t *out_data = output.data;

    // Request the first row of scale matrix
    transfer_scale(scale_data, 0);

    // Run the loop over the rows
    for (uint32_t i = 0; i < scale_data->height; i++) {
        // Wait for the scale data to be ready
        int32_t *scale = get_scale(scale_data, i);

        // Start the transfer of the next row (will be ignored if last row)
        transfer_scale(scale_data, i + 1);

        // Offset of row in input and output
        uint32_t i_off = i * scale_data->width;

        // Go through the row
        for (uint32_t j = 0; j < scale_data->width; j++) {

            // Go through and sum the inputs
            int32_t acc = 0;
            for (uint32_t k = 0; k < n_inputs; k++) {
                int32_t *in_data = input[k].data;
                acc += in_data[i_off + j];
            }
            out_data[i_off + j] = __stdfix_smul_k(acc, scale[j]);
        }
    }
}

const component_t scale_matrix = {
        .init = scale_matrix_init,
        .func = scale_matrix_exec,
        .deinit = scale_common_deinit,
        .dma_complete = scale_common_dma_complete
};

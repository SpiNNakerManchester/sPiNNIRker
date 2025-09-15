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

//! The configuration passed to the component
typedef struct {
    //! The width of the weight matrix
    //! (consequently the width of the output)
    uint32_t weights_width;
    //! The height of the weight matrix
    //! (consequently the width of the input)
    uint32_t weights_height;
    //! The height of the input / output
    uint32_t io_height;
    //! The weights (weights_width * weights_height in size)
    int32_t weights[];
} linear_matrix_config_t;

typedef struct {
    //! The width of the weight matrix
    //! (consequently the width of the output)
    union {
        uint32_t weights_width;
        uint32_t output_width;
    };
    //! The height of the weight matrix
    //! (consequently the width of the input)
    union {
        uint32_t weights_height;
        uint32_t input_width;
    };
    //! The height of the input / output
    union {
        uint32_t input_height;
        uint32_t output_height;
    };
    //! The index of this component
    uint32_t component_index: 30;
    //! Whether the weights are in SDRAM or not
    uint32_t weights_in_sdram: 1;
    //! Whether a DMA is in progress or not
    uint32_t dma_in_progress: 1;
    //! The weights; might be in SDRAM if not sufficient space
    int32_t *weights;
    //! A space to read data into, but only if in SDRAM
    int32_t *local_data[2];
    //! The index of local data to read from
    uint16_t read_index;
    //! The index of local data to write to
    uint16_t write_index;
} linear_matrix_data_t;

static void* linear_matrix_init(uint32_t index, void *params) {
    // Cast the parameters to the configuration
    linear_matrix_config_t *config = params;

    linear_matrix_data_t *data = spin1_malloc(sizeof(linear_matrix_data_t));
    if (!data) {
        log_error("Failed to allocate linear matrix data structure");
        return (void *) 0;
    }

    data->weights_width = config->weights_width;
    data->weights_height = config->weights_height;
    data->input_height = config->io_height;
    data->dma_in_progress = 0;
    data->read_index = 0;
    data->write_index = 0;
    data->component_index = index;

    // Try the weights in DTCM
    uint32_t w_sz = data->weights_width * data->weights_height * sizeof(uint32_t);
    data->weights = spin1_malloc(w_sz);
    if (!data->weights) {
        // Need to keep in SDRAM
        data->weights = config->weights;
        data->weights_in_sdram = 1;
    } else {
        // Copy to DTCM
        spin1_memcpy(data->weights, config->weights, w_sz);
        data->weights_in_sdram = 0;
    }

    return data;
}

static void transfer_weights(linear_matrix_data_t *data, uint32_t row) {
    // Get the data structure
    linear_matrix_data_t *linear_data = data;

    // If weights are in DTCM, or the row is too big, do nothing
    if (!linear_data->weights_in_sdram || row >= linear_data->weights_height) {
        return;
    }

    // Start the DMA of the row
    linear_data->dma_in_progress = 1;
    dma_id_t dma_id = {
            .index = linear_data->component_index,
            .is_component = 1,
            .is_input = 0
    };
    int32_t *weights = &linear_data->weights[row * linear_data->weights_width];
    uint32_t size = linear_data->weights_width * sizeof(int32_t);
    spin1_dma_transfer(dma_id.id, (void *) weights,
            linear_data->local_data[linear_data->write_index], DMA_READ, size);

    // The next read index is the current write index
    linear_data->read_index = linear_data->write_index;
    linear_data->write_index = (linear_data->write_index + 1) % 2;
}

static int32_t *get_weights(linear_matrix_data_t *data, uint32_t row) {
    // Get the data structure
    linear_matrix_data_t *linear_data = data;

    // If weights are in DTCM, return a pointer to the row
    if (!linear_data->weights_in_sdram) {
        return &linear_data->weights[row * linear_data->weights_width];
    }

    // If the DMA is in progress, wait for it
    uint32_t cspr = spin1_int_disable();
    while (linear_data->dma_in_progress) {
        spin1_wfi();
    }
    spin1_mode_restore(cspr);

    // Return where we need to read from
    return linear_data->local_data[linear_data->read_index];
}

static void linear_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {

    // Get the data structure
    linear_matrix_data_t *linear_data = data;

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
                    sum += ((int32_t *) input[idx].data)[i_off_in + k];
                }

                // Add the product of input sum and weight to the output
                out_data[i_off_out + j] = __stdfix_smul_k(sum, weights[j]);
            }
        }
    }
}

static void linear_matrix_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    linear_matrix_data_t *linear_data = data;
    linear_data->dma_in_progress = 0;
}

static void linear_matrix_deinit(void *data) {
    sark_free(data);
}

const component_t linear_matrix = {
    .init = linear_matrix_init,
    .func = linear_matrix_exec,
    .deinit = linear_matrix_deinit,
    .dma_complete = linear_matrix_dma_complete
};

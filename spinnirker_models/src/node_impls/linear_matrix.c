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
    uint32_t weights_width;
    //! The height of the weight matrix
    //! (consequently the width of the input)
    uint32_t weights_height;
    //! The height of the input / output
    uint32_t io_height;
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
    data->io_height = config->io_height;
    data->dma_in_progress = 0;
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

    if (!linear_data->weights_in_sdram) {
        return;
    }
}

static int32_t *get_row(linear_matrix_data_t *data, uint32_t row) {

}

static void linear_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {

    // Get the data structure
    linear_matrix_data_t *linear_data = data;

    // Convert to right type for output (Accum but only ever assigned to)
    // and clear
    int32_t *out_data = output.data;
    for (uint32_t i = 0; i < linear_data->io_height; i++) {
        uint32_t i_off = i * linear_data->weights_width;
        for (uint32_t j = 0; j < linear_data->weights_width; j++) {
            out_data[i_off + j] = 0;
        }
    }

    // For each input, do a matrix multiplication to the output
    for (uint32_t idx = 0; idx < n_inputs; idx++) {
        int32_t *input_data = input[idx].data;
        for (uint32_t i = 0; i < linear_data->io_height; i++) {
            // Offset of row i in input
            uint32_t i_off_in = i * linear_data->weights_height;
            // Offset of row i in output
            uint32_t i_out_off = i * linear_data->weights_width;
            for (uint32_t j = 0; j < linear_data->weights_width; j++) {
                // Accum as int32 - fine because we only ever add to it
                int32_t sum = 0;
                for (uint32_t k = 0; k < linear_data->weights_height; k++) {
                    uint32_t ik = i_off_in + k;
                    // Offset of row k in weights
                    uint32_t k_off = k * linear_data->weights_width;
                    uint32_t kj = k_off + j;
                    sum += __stdfix_smul_k(
                            input_data[ik], linear_data->weights[kj]);
                }

                // Write to the output
                out_data[i_out_off + j] += sum;
            }
        }
    }
}

static void linear_matrix_dma_complete(dma_id_t id, void *data) {
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

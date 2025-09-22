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
//! \file linear_common.h
//! \brief Linear NIR common implementation

#include <stdint.h>

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
} linear_common_config_t;

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
    uint32_t component_index : 30;
    //! Whether the weights are in SDRAM or not
    uint32_t weights_in_sdram : 1;
    //! Whether a DMA is in progress or not
    uint32_t dma_in_progress : 1;
    //! The weights; might be in SDRAM if not sufficient space
    int32_t *weights;
    //! A space to read data into, but only if in SDRAM
    int32_t *local_data[2];
    //! The index of local data to read from
    uint16_t read_index;
    //! The index of local data to write to
    uint16_t write_index;
} linear_common_data_t;

static linear_common_data_t *linear_common_init(uint32_t index,
        linear_common_config_t *config, linear_common_data_t *data) {
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

static void transfer_weights(linear_common_data_t *linear_data, uint32_t row) {
    // If weights are in DTCM, or the row is too big, do nothing
    if (!linear_data->weights_in_sdram || row >= linear_data->weights_height) {
        return;
    }

    // Start the DMA of the row
    linear_data->dma_in_progress = 1;
    dma_id_t dma_id = {
            .index = linear_data->component_index,
            .is_component = 1,
            .is_input = 0};
    int32_t *weights = &linear_data->weights[row * linear_data->weights_width];
    uint32_t size = linear_data->weights_width * sizeof(int32_t);
    spin1_dma_transfer(dma_id.id, (void *)weights,
            linear_data->local_data[linear_data->write_index], DMA_READ, size);

    // The next read index is the current write index
    linear_data->read_index = linear_data->write_index;
    linear_data->write_index = (linear_data->write_index + 1) % 2;
}

static int32_t *get_weights(linear_common_data_t *linear_data, uint32_t row) {
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

static void linear_common_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    linear_common_data_t *linear_data = data;
    linear_data->dma_in_progress = 0;
}

static void linear_common_deinit(void *data) {
    sark_free(data);
}

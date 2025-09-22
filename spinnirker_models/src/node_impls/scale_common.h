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
//! \file scale_common.h
//! \brief scale NIR common implementation

#include <stdint.h>
#include <spin1_api.h>

//! The configuration passed to the component
typedef struct {
    //! The width of the scale matrix, input and output
    uint32_t width;
    //! The height of the scale matrix, input and output
    uint32_t height;
    //! The scale matrix data (size width*height)
    int32_t scale[];
} scale_common_config_t;

typedef struct {
    //! The width of the weights, input and output
    uint32_t width;
    //! The height of the weights, input and output
    uint32_t height;
    //! The index of this component
    uint32_t component_index : 30;
    //! Whether the weights are in SDRAM or not
    uint32_t weights_in_sdram : 1;
    //! Whether a DMA is in progress or not
    uint32_t dma_in_progress : 1;
    //! The scale matrix; might be in SDRAM if not sufficient space
    int32_t *scale;
    //! A space to read data into, but only if in SDRAM
    int32_t *local_data[2];
    //! The index of local data to read from
    uint16_t read_index;
    //! The index of local data to write to
    uint16_t write_index;
} scale_common_data_t;

static scale_common_data_t *scale_common_init(uint32_t index,
        scale_common_config_t *config, scale_common_data_t *data) {
    data->width = config->width;
    data->height = config->height;
    data->dma_in_progress = 0;
    data->read_index = 0;
    data->write_index = 0;
    data->component_index = index;

    // Try the scale matrix in DTCM
    uint32_t sz = data->width * data->height * sizeof(int32_t);
    data->scale = spin1_malloc(sz);
    if (!data->scale) {
        // Need to keep in SDRAM
        data->scale = config->scale;
        data->weights_in_sdram = 1;
    } else {
        // Copy to DTCM
        spin1_memcpy(data->scale, config->scale, sz);
        data->weights_in_sdram = 0;
    }
    return data;
}

static void transfer_scale(scale_common_data_t *scale_data, uint32_t row) {
    // If scale matrix is in DTCM, or the row is too big, do nothing
    if (!scale_data->weights_in_sdram || row >= scale_data->height) {
        return;
    }

    // Start the DMA of the row
    scale_data->dma_in_progress = 1;
    dma_id_t dma_id = {
            .index = scale_data->component_index,
            .is_component = 1,
            .is_input = 0};
    int32_t *weights = &scale_data->scale[row * scale_data->width];
    uint32_t size = scale_data->width * sizeof(int32_t);
    spin1_dma_transfer(dma_id.id, (void *)weights,
            scale_data->local_data[scale_data->write_index], DMA_READ, size);

    // The next read index is the current write index
    scale_data->read_index = scale_data->write_index;
    scale_data->write_index = (scale_data->write_index + 1) % 2;
}

static int32_t *get_scale(scale_common_data_t *scale_data, uint32_t row) {
    // If scale matrix is in DTCM, return a pointer to the row
    if (!scale_data->weights_in_sdram) {
        return &scale_data->scale[row * scale_data->width];
    }

    // If the DMA is in progress, wait for it
    uint32_t cspr = spin1_int_disable();
    while (scale_data->dma_in_progress) {
        spin1_wfi();
    }
    spin1_mode_restore(cspr);

    // Return where we need to read from
    return scale_data->local_data[scale_data->read_index];
}

static void scale_common_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    scale_common_data_t *scale_data = data;
    scale_data->dma_in_progress = 0;
}

static void scale_common_deinit(void *data) {
    sark_free(data);
}

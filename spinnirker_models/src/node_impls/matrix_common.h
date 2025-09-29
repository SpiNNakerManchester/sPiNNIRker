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
//! \file matrix_common.h
//! \brief Common matrix functions

#ifndef _MATRIX_COMMON_H_
#define _MATRIX_COMMON_H_

#include <stdint.h>
#include <spin1_api.h>
#include <debug.h>
#include "../component.h"

typedef struct {
    //! The width of the matrix of data
    uint32_t width;
    //! The height of the matrix of data
    uint32_t height;
    //! The data values (width * height in size)
    int32_t data[];
} matrix_config_t;

typedef struct {
    //! The width of the matrix of data
    uint32_t width;
    //! The height of the matrix of data
    uint32_t height;
    //! The index of the component for DMA identification
    uint32_t component_index : 30;
    //! Whether the data is in SDRAM or not
    uint32_t in_sdram : 1;
    //! Whether a DMA is in progress or not
    uint32_t dma_in_progress : 1;
    //! A space to read data into, but only if in SDRAM
    int32_t *local_data[2];
    //! The index of local data to read from
    uint32_t read_index;
    //! The index of local data to write to
    uint32_t write_index;
    //! The matrtix data values; might be in SDRAM if not sufficient space
    int32_t *data;
} matrix_data_t;

static matrix_data_t *matrix_init(uint32_t index, matrix_config_t *config,
        matrix_data_t *data) {
    data->width = config->width;
    data->height = config->height;
    data->dma_in_progress = 0;
    data->read_index = 0;
    data->write_index = 0;
    data->component_index = index;

    // Try the data in DTCM
    uint32_t sz = data->width * data->height * sizeof(int32_t);
    data->data = spin1_malloc(sz);
    if (!data->data) {
        // Need to keep in SDRAM
        data->data = config->data;
        data->in_sdram = 1;
    } else {
        // Copy to DTCM
        spin1_memcpy(data->data, config->data, sz);
        data->in_sdram = 0;
    }
    return data;
}

static void matrix_transfer_row(matrix_data_t *matrix_data, uint32_t row) {
    // If data is in DTCM, or the row is too big, do nothing
    if (!matrix_data->in_sdram || row >= matrix_data->height) {
        return;
    }

    // Start the DMA of the row
    matrix_data->dma_in_progress = 1;
    dma_id_t dma_id = {
            .index = matrix_data->component_index,
            .is_component = 1,
            .is_input = 0};
    int32_t *weights = &matrix_data->data[row * matrix_data->width];
    uint32_t size = matrix_data->width * sizeof(int32_t);
    spin1_dma_transfer(dma_id.id, (void *) weights,
            matrix_data->local_data[matrix_data->write_index], DMA_READ, size);

    // The next read index is the current write index
    matrix_data->read_index = matrix_data->write_index;
    matrix_data->write_index = (matrix_data->write_index + 1) % 2;
}

static int32_t *matrix_get_row(matrix_data_t *matrix_data, uint32_t row) {
    // If data is in DTCM, return a pointer to the row
    if (!matrix_data->in_sdram) {
        return &matrix_data->data[row * matrix_data->width];
    }

    // If the DMA is in progress, wait for it
    uint32_t cspr = spin1_int_disable();
    while (matrix_data->dma_in_progress) {
        spin1_wfi();
    }
    spin1_mode_restore(cspr);

    // Return where we need to read from
    return matrix_data->local_data[matrix_data->read_index];
}

static void matrix_clear_outputs(data_t output, uint32_t n_words) {
    // Convert to right type for output (Accum but only ever added to)
    uint32_t *out_data = output.data;

    // Clear the output
    for (uint32_t i = 0; i < n_words; i++) {
        out_data[i] = 0;
    }
}

static void matrix_common_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    matrix_data_t *matrix_data = data;
    matrix_data->dma_in_progress = 0;
}

#endif // _MATRIX_COMMON_H_

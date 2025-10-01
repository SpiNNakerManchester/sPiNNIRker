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
//! \file matrix_common_init.c
//! \brief Common matrix initialisation functions

#include <spin1_api.h>
#include <arm_acle.h>
#include <debug.h>
#include "../component.h"
#include "matrix_common_init.h"

matrix_data_t *matrix_init(uint32_t index, matrix_config_t *config,
        matrix_data_t *data, uint32_t element_size) {
    data->width = config->width;
    data->height = config->height;
    data->element_size = element_size;
    data->dma_in_progress = 0;
    data->read_index = 0;
    data->write_index = 0;
    data->component_index = index;

    // Try the data in DTCM
    uint32_t sz = data->width * data->height * data->element_size;
    data->data = spin1_malloc(sz);
    if (!data->data) {
        // Need to keep in SDRAM
        log_warning("Matrix component %u data too large for DTCM, "
            "keeping in SDRAM", index);
        data->data = config->data;
        data->in_sdram = 1;

        // This means we need to allocate two rows of local data for transfers
        uint32_t row_sz = data->width * data->element_size;
        for (uint32_t i = 0; i < 2; i++) {
            data->local_data[i] = spin1_malloc(row_sz);
            if (!data->local_data[i]) {
                log_error("Failed to allocate local data %i for matrix "
                    "component %u", i, index);
                return NULL;
            }
        }
    } else {
        // Copy to DTCM
        spin1_memcpy(data->data, config->data, sz);
        data->in_sdram = 0;
    }
    return data;
}

void matrix_common_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    matrix_data_t *matrix_data = data;
    matrix_data->dma_in_progress = 0;
}

void matrix_transfer_row(matrix_data_t *matrix_data, uint32_t row) {
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
    uint32_t *weights = &matrix_data->data[row * matrix_data->width *
            matrix_data->element_size];
    uint32_t size = matrix_data->width * matrix_data->element_size;
    spin1_dma_transfer(dma_id.id, (void *) weights,
            matrix_data->local_data[matrix_data->write_index], DMA_READ, size);

    // The next read index is the current write index
    matrix_data->read_index = matrix_data->write_index;
    matrix_data->write_index = (matrix_data->write_index + 1) % 2;
}

void *matrix_get_row(matrix_data_t *matrix_data, uint32_t row) {
    // If data is in DTCM, return a pointer to the row
    if (!matrix_data->in_sdram) {
        return &matrix_data->data[row * matrix_data->width *
            matrix_data->element_size];
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

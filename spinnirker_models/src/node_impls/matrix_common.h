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
#include <arm_acle.h>

#include "../component.h"
#include "matrix_common_init.h"

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

static void matrix_common_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    matrix_data_t *matrix_data = data;
    matrix_data->dma_in_progress = 0;
}

#endif // _MATRIX_COMMON_H_

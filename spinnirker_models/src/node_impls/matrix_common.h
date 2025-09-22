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

#include <stdint.h>
#include <spin1_api.h>
#include <debug.h>
#include "../component.h"

static void matrix_transfer_row(uint32_t width, uint32_t component_index,
    int32_t *sdram_data, int32_t **local_data, uint32_t *write_index,
    uint32_t *read_index, uint32_t row) {

    // Start the DMA of the row
    dma_id_t dma_id = {
            .index = component_index,
            .is_component = 1,
            .is_input = 0};
    int32_t *weights = &sdram_data[row * width];
    uint32_t size = width * sizeof(int32_t);
    spin1_dma_transfer(dma_id.id, (void *) weights,
            local_data[*write_index], DMA_READ, size);

    // The next read index is the current write index
    *read_index = *write_index;
    *write_index = (*write_index + 1) % 2;
}

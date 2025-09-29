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

#include <spin1_api.h>
#include "matrix_common_init.h"

matrix_data_t *matrix_init(uint32_t index, matrix_config_t *config,
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
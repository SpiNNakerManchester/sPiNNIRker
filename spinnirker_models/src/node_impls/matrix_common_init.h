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
//! \file matrix_common_init.h
//! \brief Common matrix initialization functions
#include <stdint.h>

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

extern matrix_data_t *matrix_init(uint32_t index, matrix_config_t *config,
        matrix_data_t *data);

extern void matrix_common_dma_complete(UNUSED dma_id_t id, void *data);

extern void matrix_transfer_row(matrix_data_t *matrix_data, uint32_t row);

extern int32_t *matrix_get_row(matrix_data_t *matrix_data, uint32_t row);

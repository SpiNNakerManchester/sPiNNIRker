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
//! \file matrix_matrix_common.h
//! \brief Common matrix functions when the input is matrices

#include "matrix_common.h"

typedef struct {
    //! The information about the matrix being processed
    matrix_data_t *data;

    //! The current index of the row being processed
    uint32_t current_row;

    //! The current index of the column being processed
    uint32_t current_col;

    //! The current row of the matrix that has been transferred
    int32_t *current_data;
} matrix_loop_t;

static matrix_loop_t matrix_loop_start(matrix_data_t *matrix) {
    matrix_loop_t loop;
    loop.data = matrix;
    loop.current_row = 0;
    loop.current_col = 0;

    // Start the transfer of the first row
    matrix_transfer_row(loop.data, 0);

    return loop;
}

static uint32_t matrix_loop_is_next(matrix_loop_t *loop, int32_t *value,
        uint32_t *row, uint32_t *col) {
    // Loop complete!
    if (loop->current_row >= loop->data->height) {
        return 0;
    }

    // If we are in column 0, get the data for the row
    if (loop->current_col == 0) {
        // Get the current row data
        loop->current_data = matrix_get_row(loop->data, loop->current_row);


        // Start the transfer for the next row (ignored if last row)
        matrix_transfer_row(loop->data, loop->current_row + 1);
    }

    // Get the value
    *value = loop->current_data[loop->current_col];
    *row = loop->current_row;
    *col = loop->current_col;

    // Move to the next column
    loop->current_col++;
    if (loop->current_col >= loop->data->width) {
        // Move to the next row
        loop->current_col = 0;
        loop->current_row++;
    }

    // Return that we have a value
    return 1;
}

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
//! \file matrix_spikes_common.h
//! \brief Common functions when the input is spikes

#include "matrix_common.h"

typedef struct {

    //! The inputs
    data_t *inputs;

    //! The number of input spike lists
    uint32_t n_inputs;

    //! The input width
    uint32_t input_width;

    //! The inverse of the input width, to do division by
    div_const input_width_inv;

    //! The current input spike list index
    uint32_t current_input_index;

    //! The current spike index in the current input spike list
    uint32_t current_spike_index;

    //! The next row value
    uint32_t next_row;

    //! The next column value
    uint32_t next_col;

    //! A pointer to the next row of the matrix to transfer.  Like transfer_row,
    //! this can be either the next_row or the next_column depending on the
    //! setup.
    uint32_t *next_transfer_row;

    //! A pointer to the index of the next value to read from the current_data
    //! array.  This is either the next_row or next_column depending on the
    //! setup.
    uint32_t *value_index;

    //! Whether there is a next spike or not
    uint32_t is_next;

    //! The current row of the matrix that has been transferred
    void *current_data;

    //! The matrix data being processed
    matrix_data_t *matrix_data;
} matrix_spikes_loop_data_t;

extern matrix_spikes_loop_data_t matrix_spikes_loop_start(
        data_t *inputs, uint32_t n_inputs, uint32_t input_width,
        div_const inv_input_width, uint32_t transpose,
        matrix_data_t *matrix_data);

extern uint32_t matrix_spikes_loop_is_next(matrix_spikes_loop_data_t *loop,
        uint32_t *row, uint32_t *col);

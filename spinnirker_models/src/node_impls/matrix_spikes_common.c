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
//! \file matrix_spikes_common.c
//! \brief Common functions when the input is spikes

#include "matrix_spikes_common.h"

static uint32_t matrix_spikes_is_next(matrix_spikes_loop_data_t *loop,
        uint32_t *spike) {
    // A loop, but will only cycle twice at most...
    // Only even look if there is still a valid input list to look in
    while (loop->current_input_index < loop->n_inputs) {
        // There is still a valid list, so see if there is a spike in it
        spike_list_t *spikes = loop->inputs[loop->current_input_index].data;
        if (loop->current_spike_index >= spikes->n_spikes) {
            // This list is done, so move to the next one if there is one
            loop->current_input_index++;
            loop->current_spike_index = 0;
        } else {
            // There is a spike, so get it, move on, and return it
            *spike = spikes->spikes[loop->current_spike_index].global_source_id;
            loop->current_spike_index++;
            return 1;
        }
    }
    return 0;
}

static uint32_t matrix_spikes_is_next_row_col(matrix_spikes_loop_data_t *loop,
        uint32_t input_width, div_const input_width_inv,
        uint32_t *row, uint32_t *col) {
    uint32_t spike;
    if (matrix_spikes_is_next(loop, &spike)) {
        // Decode the spike into row and column
        *row = div_by_const(spike, input_width_inv);
        *col = spike - (*row * input_width);
        return 1;
    }
    return 0;
}

matrix_spikes_loop_data_t matrix_spikes_loop_start(
        data_t *inputs, uint32_t n_inputs, uint32_t input_width,
        div_const inv_input_width, uint32_t transpose,
        matrix_data_t *matrix_data) {
    matrix_spikes_loop_data_t loop;
    loop.inputs = inputs;
    loop.n_inputs = n_inputs;
    loop.input_width = input_width;
    loop.input_width_inv = inv_input_width;
    loop.matrix_data = matrix_data;
    loop.current_input_index = 0;
    loop.current_spike_index = 0;
    loop.next_row = 0;
    loop.next_col = 0;
    loop.next_transfer_row = transpose ? &loop.next_col : &loop.next_row;
    loop.value_index = transpose ? &loop.next_row : &loop.next_col;
    loop.is_next = 0;

    // Get the first spike if there is one
    loop.is_next = matrix_spikes_is_next_row_col(&loop, loop.input_width,
            loop.input_width_inv, &loop.next_row, &loop.next_col);

    // If there is a next spike, start transferring it
    if (loop.is_next) {
        matrix_transfer_row(matrix_data, *loop.next_transfer_row);
    }
    return loop;
}

uint32_t matrix_spikes_loop_is_next(matrix_spikes_loop_data_t *loop,
        uint32_t *row, uint32_t *col) {
    if (!loop->is_next) {
        return 0;
    }

    // Get the current data for the row
    // Note we ignore maybe-uninitialized here as the static analysis can't see
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    loop->current_data = matrix_get_row(loop->matrix_data,
        *loop->next_transfer_row);
    *row = loop->next_row;
    *col = loop->next_col;
    uint32_t last_transfer_row = *loop->next_transfer_row;

    // Get the next spike if there is one
    loop->is_next = matrix_spikes_is_next_row_col(loop, loop->input_width,
            loop->input_width_inv, &loop->next_row, &loop->next_col);
    if (loop->is_next) {
        // If the next spike is different, transfer it
        if (*loop->next_transfer_row != last_transfer_row) {
            matrix_transfer_row(loop->matrix_data, *loop->next_transfer_row);
        }
    }
    return 1;
}

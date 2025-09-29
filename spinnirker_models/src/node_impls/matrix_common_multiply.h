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
//! \file matrix_matrix_common_multiply.h
//! \brief Common function to do the matrix multiplication when the input is
//!        matrices

#include <stdfix-full-iso.h>
#include "matrix_common.h"

static void matrix_matrix_multiply(matrix_data_t *matrix_data,
        uint32_t row, uint32_t col, int32_t *current_data,
        data_t *input, uint32_t n_inputs, int32_t *out_data) {
    // Go through column k = matrix row of each of the input rows
    // (which is each of the matrix columns, hence we loop over width)
    for (uint32_t i = 0; i < matrix_data->width; i++) {
        // Offset of row i in input (which is same width as matrix height)
        uint32_t i_off_in = i * matrix_data->height;
        // Offset of row i in output (which is same width as matrix width)
        uint32_t i_off_out = i * matrix_data->width;

        // Add up each of the inputs
        int32_t sum = 0;
        for (uint32_t idx = 0; idx < n_inputs; idx++) {
            sum += ((int32_t *)input[idx].data)[i_off_in + row];
        }

        // Add the product of input sum and weight to the output
        out_data[i_off_out + col] = __stdfix_smul_k(sum, current_data[col]);
    }
}

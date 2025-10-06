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
//! \file current_based_leaky_integrator_matrix.c
//! \brief leaky_integrator NIR component implementation (matrix inputs)

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "current_based_leaky_integrator_matrix.h"
#include "matrix_matrix_common.h"
#include "decay.h"

typedef struct {
    int32_t current;         //!< The current value - signed long accum
    uint32_t current_decay;  //!< The current decay factor - unsigned long fract
    int32_t current_weight;  //!< The current weight - signed long accum
    uint32_t decay;          //!< The decay factor - unsigned long fract
    int32_t resistance;      //!< The resistance factor - signed long accum
} leaky_integrator_item_t;

static void *current_based_leaky_integrator_matrix_init(uint32_t index, void *params) {
    matrix_data_t *data = spin1_malloc(sizeof(matrix_data_t));
    if (!data) {
        log_error("Failed to allocate leaky_integrator matrix data structure");
        return (void *)0;
    }
    return matrix_init(index, params, data, sizeof(leaky_integrator_item_t));
}

static void current_based_leaky_integrator_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    matrix_data_t *leaky_integrator_data = data;

    matrix_loop_t loop = matrix_loop_start(leaky_integrator_data);
    int32_t *out_data = output.data;

    uint32_t row;
    uint32_t col;
    while (matrix_loop_is_next(&loop, &row, &col)) {
        leaky_integrator_item_t *row_data = loop.current_data;
        // Go through and sum the inputs
        int32_t acc = 0;
        uint32_t i_off = row * leaky_integrator_data->width;
        for (uint32_t k = 0; k < n_inputs; k++) {
            int32_t *in_data = input[k].data;
            acc += in_data[i_off + col];
        }
        // Decay the current and add the input
        row_data[col].current = decay(row_data[col].current,
                row_data[col].current_decay);
        row_data[col].current += __stdfix_smul_k(acc,
                row_data[col].current_weight);
        // Decay the output and add the current scaled by resistance
        out_data[i_off + col] = decay(out_data[i_off + col],
                row_data[col].decay);
        out_data[i_off + col] += __stdfix_smul_k(row_data[col].current,
                row_data[col].resistance);
    }
}

const component_t current_based_leaky_integrator_matrix = {
        .init = current_based_leaky_integrator_matrix_init,
        .func = current_based_leaky_integrator_matrix_exec,
        .dma_complete = matrix_common_dma_complete
};

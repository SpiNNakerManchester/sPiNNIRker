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
//! \file integrate_and_fire_matrix.c
//! \brief integrate_and_fire NIR component implementation (matrix inputs)

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "integrate_and_fire_matrix.h"
#include "matrix_matrix_common.h"
#include "spike_output_common.h"

typedef struct {
    int32_t resistance;  //!< The resistance factor - signed long accum
    int32_t threshold;   //!< The firing threshold - signed long accum
    int32_t reset;       //!< The reset voltage - signed long accum
    int32_t voltage;     //!< The current voltage - signed long accum
} leaky_integrate_and_fire_item_t;

typedef struct {
    uint32_t key;
    matrix_config_t matrix_data;
} integrate_and_fire_matrix_config_t;

typedef struct {
    uint32_t key;
    matrix_data_t matrix_data;
} integrate_and_fire_matrix_data_t;

static void *integrate_and_fire_matrix_init(uint32_t index, void *params) {
    integrate_and_fire_matrix_data_t *data = spin1_malloc(
            sizeof(integrate_and_fire_matrix_data_t));
    if (!data) {
        log_error("Failed to allocate leaky_integrate_and_fire matrix data structure");
        return (void *)0;
    }
    integrate_and_fire_matrix_config_t *config = params;
    data->key = config->key;
    return matrix_init(index, &config->matrix_data, &data->matrix_data,
            sizeof(leaky_integrate_and_fire_item_t));
}

static void integrate_and_fire_matrix_exec(void *data, uint32_t n_inputs,
            data_t *input, data_t output) {
    // Get the data structure
    integrate_and_fire_matrix_data_t *lif_data = data;
    matrix_data_t *matrix_data = &lif_data->matrix_data;

    matrix_loop_t loop = matrix_loop_start(matrix_data);
    spike_list_t *out_data = output.data;
    out_data->n_spikes = 0;

    uint32_t row;
    uint32_t col;
    while (matrix_loop_is_next(&loop, &row, &col)) {
        leaky_integrate_and_fire_item_t *row_data = loop.current_data;
        // Go through and sum the inputs
        int32_t acc = 0;
        uint32_t i_off = row * matrix_data->width;
        for (uint32_t k = 0; k < n_inputs; k++) {
            int32_t *in_data = input[k].data;
            acc += in_data[i_off + col];
        }
        row_data[col].voltage += __stdfix_smul_k(acc, row_data[col].resistance);
        if (row_data[col].voltage >= row_data[col].threshold) {
            spike(out_data, lif_data->key, i_off + col);
            row_data[col].voltage = row_data[col].reset;
        }
    }
}

const component_t integrate_and_fire_matrix = {
        .init = integrate_and_fire_matrix_init,
        .func = integrate_and_fire_matrix_exec,
        .dma_complete = matrix_common_dma_complete
};

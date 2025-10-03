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
//! \file leaky_integrate_and_fire_spikes.c
//! \brief leaky_integrate_and_fire NIR component implementation (matrix inputs)

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "leaky_integrate_and_fire_spikes.h"
#include "matrix_spikes_common.h"
#include "decay_multi.h"
#include "spike_output_common.h"

typedef struct {
    uint32_t last_time;  //!< The last time the neuron was updated
    uint32_t decay;      //!< The decay factor - unsigned long accum
    int32_t resistance;  //!< The resistance factor - signed long accum
    int32_t threshold;   //!< The firing threshold - signed long accum
    int32_t reset;       //!< The reset voltage - signed long accum
    int32_t voltage;     //!< The current voltage - signed long accum
} leaky_integrate_and_fire_spikes_item_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;
    uint32_t key;
    matrix_config_t matrix_data;
} leaky_integrate_and_fire_spikes_config_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;
    uint32_t key;
    matrix_data_t matrix_data;
    uint32_t time;
} leaky_integrate_and_fire_spikes_data_t;

static void *leaky_integrate_and_fire_spikes_init(uint32_t index, void *params) {
    leaky_integrate_and_fire_spikes_data_t *data = spin1_malloc(
            sizeof(leaky_integrate_and_fire_spikes_data_t));
    if (!data) {
        log_error("Failed to allocate leaky_integrate_and_fire spikes data structure");
        return (void *)0;
    }
    leaky_integrate_and_fire_spikes_config_t *config = params;
    data->input_width_inv = config->input_width_inv;
    data->key = config->key;
    data->time = 0;
    matrix_data_t *matrix_data = matrix_init(index, &config->matrix_data,
            &data->matrix_data, sizeof(leaky_integrate_and_fire_spikes_item_t));
    leaky_integrate_and_fire_spikes_item_t *items =
        (leaky_integrate_and_fire_spikes_item_t *) matrix_data->data;
    for (uint32_t i = 0; i < matrix_data->width * matrix_data->height; i++) {
        items[i].last_time = 0;
    }
    return data;
}

static void leaky_integrate_and_fire_spikes_exec(void *data, uint32_t n_inputs,
            data_t *input, data_t output) {
    // Get the data structure
    leaky_integrate_and_fire_spikes_data_t *lif_data = data;
    matrix_data_t *matrix_data = &lif_data->matrix_data;

    matrix_spikes_loop_data_t loop = matrix_spikes_loop_start(input, n_inputs,
        matrix_data->width, lif_data->input_width_inv, 0, matrix_data);
    spike_list_t *out_data = output.data;
    out_data->n_spikes = 0;

    uint32_t row;
    uint32_t col;
    while (matrix_spikes_loop_is_next(&loop, &row, &col)) {
        leaky_integrate_and_fire_spikes_item_t *row_data = loop.current_data;
        uint32_t i_off = row * matrix_data->width;
        row_data[col].voltage = decay_multi(row_data[col].voltage,
            row_data[col].decay, lif_data->time - row_data[col].last_time);
        row_data[col].last_time = lif_data->time;
        row_data[col].voltage += row_data[col].resistance;
        if (row_data[col].voltage >= row_data[col].threshold) {
            spike(out_data, lif_data->key, i_off + col);
            row_data[col].voltage = row_data[col].reset;
        }
    }
    lif_data->time++;
}

const component_t leaky_integrate_and_fire_spikes = {
        .init = leaky_integrate_and_fire_spikes_init,
        .func = leaky_integrate_and_fire_spikes_exec,
        .dma_complete = matrix_common_dma_complete
};

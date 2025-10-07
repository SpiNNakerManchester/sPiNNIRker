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
//! \file current_based_leaky_integrate_and_fire_spikes.c
//! \brief leaky_integrator NIR component implementation (spike inputs)
//! This is a 2D matrix multiplication of inputs by weights, but using
//! spikes for improved efficiency.

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "current_based_leaky_integrate_and_fire_spikes.h"
#include "matrix_spikes_common.h"
#include "matrix_matrix_common.h"
#include "decay.h"
#include "spike_output_common.h"

typedef struct {
    int32_t current;         //!< The current value - signed long accum
    uint32_t current_decay;  //!< The current decay factor - unsigned long fract
    int32_t current_weight;  //!< The current weight - signed long accum
    uint32_t decay;          //!< The voltage decay factor - unsigned long fract
    int32_t resistance;      //!< The resistance factor - signed long accum
    int32_t threshold;       //!< The firing threshold - signed long accum
    int32_t reset;           //!< The reset value - signed long accum
    int32_t voltage;         //!< The voltage value - signed long accum
} cuba_lif_spikes_item_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    //! The key for output spikes
    uint32_t key;

    matrix_config_t matrix;
} cuba_lif_spikes_config_t;

typedef struct {
    //! 1/the width of the input (to do division by)
    div_const input_width_inv;

    //! The key for output spikes
    uint32_t key;

    matrix_data_t matrix;
} cuba_lif_spikes_data_t;

static void *current_based_leaky_integrate_and_fire_spikes_init(uint32_t index, void *params) {
    cuba_lif_spikes_data_t *data = spin1_malloc(
            sizeof(cuba_lif_spikes_data_t));
    if (!data) {
        log_error("Failed to allocate leaky_integrator matrix data structure");
        return (void *)0;
    }
    cuba_lif_spikes_config_t *spikes_config = params;
    matrix_init(index, &spikes_config->matrix, &data->matrix,
        sizeof(cuba_lif_spikes_item_t));
    data->input_width_inv = spikes_config->input_width_inv;
    return data;
}

static void current_based_leaky_integrate_and_fire_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    cuba_lif_spikes_data_t *spikes_data = data;
    matrix_data_t *lif_data = &spikes_data->matrix;
    spike_list_t *out_data = output.data;
    uint32_t row;
    uint32_t col;

    // Decay the existing current values
    matrix_loop_t current_loop = matrix_loop_start(lif_data);
    while (matrix_loop_is_next(&current_loop, &row, &col)) {
        cuba_lif_spikes_item_t *row_data = current_loop.current_data;
        row_data[col].current = decay(row_data[col].current,
                row_data[col].current_decay);
    }

    // Process the spikes into the current
    matrix_spikes_loop_data_t spike_loop = matrix_spikes_loop_start(
            input, n_inputs, lif_data->width, spikes_data->input_width_inv, 0,
            lif_data);
    while (matrix_spikes_loop_is_next(&spike_loop, &row, &col)) {
        cuba_lif_spikes_item_t *row_data = spike_loop.current_data;
        // Add in the new current value to be added per spike
        row_data[col].current += row_data[col].current_weight;
    }

    // Process the voltages
    matrix_loop_t voltage_loop = matrix_loop_start(lif_data);
    while (matrix_loop_is_next(&voltage_loop, &row, &col)) {
        cuba_lif_spikes_item_t *row_data = voltage_loop.current_data;
        // Decay voltage
        row_data[col].voltage = decay(row_data[col].voltage,
                row_data[col].decay);
        // Add in current * resistance
        row_data[col].voltage += row_data[col].current * row_data[col].resistance;

        // Check for spike
        if (row_data[col].voltage >= row_data[col].threshold) {
            // Reset the voltage
            row_data[col].voltage = row_data[col].reset;
            // Send the spike event
            spike(out_data, spikes_data->key,
                row * lif_data->width + col);
        }
    }
}

const component_t current_based_leaky_integrate_and_fire_spikes = {
    .init = current_based_leaky_integrate_and_fire_spikes_init,
    .func = current_based_leaky_integrate_and_fire_spikes_exec,
    .dma_complete = matrix_common_dma_complete
};

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
//! \file output_matrix.c
//! \brief Output NIR component implementation - with spike inputs

#include "output_spikes.h"

#include <arm_acle.h>
#include <spin1_api.h>
#include <debug.h>
#include <bit_field.h>

//! The configuration passed to the output component
typedef struct {
    //! The number of words to record per input per time step
    uint32_t words_per_input;
    //! The space to write the output data to
    uint32_t data[];
} output_matrix_config_t;

typedef struct {
    //! The time of the component
    uint32_t time;
    //! How many works are needed to store spikes from all neurons per time step
    uint32_t words_per_input;
    //! Temporary storage for spikes in a time step
    bit_field_t local_data;
    //! A pointer to the data to receive the transfer
    uint32_t *data;
} output_matrix_data_t;

static void* output_spikes_init(UNUSED uint32_t index, void *params) {
    // Cast the parameters to the input configuration
    output_matrix_config_t *config = (output_matrix_config_t *) params;

    output_matrix_data_t *output_data = spin1_malloc(sizeof(output_matrix_data_t));
    if (!output_data) {
        log_error("Failed to allocate input data structure");
        return (void *) 0;
    }

    output_data->time = 0;
    output_data->words_per_input = config->words_per_input;
    output_data->data = config->data;

    // Allocate local storage for spikes in a time step
    output_data->local_data = spin1_malloc(config->words_per_input * sizeof(uint32_t));
    if (!output_data->local_data) {
        log_error("Failed to allocate local spike storage");
        return (void*) 0;
    }

    return output_data;
}

static void output_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        UNUSED data_t output) {

    // Get the output data structure
    output_matrix_data_t *output_data = data;

    // Base position in output data array is the position after all time steps
    // so far
    uint32_t base_pos = output_data->time * output_data->words_per_input * n_inputs;
    uint32_t size = output_data->words_per_input * sizeof(uint32_t);
    for (uint32_t input_index = 0; input_index < n_inputs; input_index++) {
        // Clear the local bitfield
        for (uint32_t w = 0; w < output_data->words_per_input; w++) {
            output_data->local_data[w] = 0;
        }

        // Convert spikes to bits to be recorded
        spike_list_t *spikes = (spike_list_t *) input[input_index].data;
        for (uint32_t s = 0; s < spikes->n_spikes; s++) {
            spike_t spike = spikes->spikes[s];
            bit_field_set(output_data->local_data, spike.global_source_id);
        }

        // Get the position in the output data to write to
        uint32_t pos = base_pos + (output_data->words_per_input * input_index);

        // Copy to the output data
        spin1_memcpy(&output_data->data[pos], output_data->local_data, size);

    }
    output_data->time++;
}

const component_t output_spikes = {
    .init = output_spikes_init,
    .func = output_spikes_exec
};

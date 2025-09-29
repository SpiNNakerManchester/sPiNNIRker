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
//! \brief Output NIR component implementation - with matrix inputs

#include "output_matrix.h"

#include <arm_acle.h>
#include <spin1_api.h>
#include <debug.h>

//! The configuration passed to the output component
typedef struct {
    //! How many words to be transferred from each input per timestep
    uint32_t words_per_input;
    //! The data itself
    uint32_t data[];
} output_matrix_config_t;

typedef struct {
    //! The time of the component
    uint32_t time;
    //! How many words to be transferred to the data per timestep
    uint32_t words_per_input;
    //! A pointer to the data to receive the transfer
    uint32_t *data;
} output_matrix_data_t;

static void* output_matrix_init(UNUSED uint32_t index, void *params) {
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

    return output_data;
}

static void output_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        UNUSED data_t output) {

    // Get the output data structure
    output_matrix_data_t *output_data = data;

    // TODO: Could we do this with a DMA?  Could be potential interference with
    // other DMAs in progress...
    // Base position in output data array is the position after all time steps
    // so far
    uint32_t base_pos = output_data->time * output_data->words_per_input * n_inputs;
    uint32_t size = output_data->words_per_input * sizeof(uint32_t);
    for (uint32_t input_index = 0; input_index < n_inputs; input_index++) {
        uint32_t pos = base_pos + (output_data->words_per_input * input_index);
        spin1_memcpy(&output_data->data[pos], input[input_index].data, size);
    }
    output_data->time++;
}

const component_t output_matrix = {
    .init = output_matrix_init,
    .func = output_matrix_exec
};

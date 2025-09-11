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
//! \file input.c
//! \brief Input NIR component implementation

#include "input.h"
#include <arm_acle.h>
#include <spin1_api.h>
#include <debug.h>

typedef struct {
    uint32_t n_words;
    uint32_t words[];
} next_data_t;

typedef struct {
    //! The time of the component
    uint32_t time;
    //! A pointer to the next data to be transferred
    next_data_t *data;
} input_data_t;

static void* input_init(void *params) {
    // Cast the parameters to the input configuration
    next_data_t *config = (next_data_t *) params;

    input_data_t *input_data = spin1_malloc(sizeof(input_data_t));
    if (!input_data) {
        log_error("Failed to allocate input data structure");
        return (void *) 0;
    }

    input_data->time = 0;
    input_data->data = config;

    return input_data;
}

static void input_exec(void *data, UNUSED uint32_t n_inputs, UNUSED data_t *input,
        data_t output) {

    // Get the input data structure
    input_data_t *input_data = data;

    // TODO: Could we do this with a DMA?  Could be potential interference with
    // other DMAs in progress...
    uint32_t n_words = input_data->data->n_words;
    spin1_memcpy(output.data, input_data->data, n_words * sizeof(uint32_t));

    // Move to next data and time
    input_data->time++;
    input_data->data = (next_data_t *) &input_data->data->words[n_words];
}

static void input_deinit(void *data) {
    sark_free(data);
}

const component_t input = {
    .init = input_init,
    .func = input_exec,
    .deinit = input_deinit
};

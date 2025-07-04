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

//! The configuration passed to the input component
typedef struct {
    //! How many words to be transferred from the data per timestep
    uint32_t words_per_time_step;
    //! The data itself
    uint32_t data[];
} input_config_t;

typedef struct {
    //! The time of the component
    uint32_t time;
    //! How many words to be transferred from the data per timestep
    uint32_t words_per_time_step;
    //! A pointer to the next data to be transferred
    uint32_t *data;
} input_data_t;

static void input_exec(void *data, uint32_t n_inputs, void **input, void *output) {
    // TODO: Fill in
}

static void* input_init(void *params) {
    // TODO: Fill in
    return (void *) 0;
}

static void input_deinit(void *data) {
    // TODO: Fill in
}

const component_t input = {
    .func = input_exec,
    .init = input_init,
    .deinit = input_deinit
};

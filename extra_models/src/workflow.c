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
//! \file workflow.h
//! \brief Workflow execution

#include "workflow.h"
#include <spin1_api.h>

//! A list of components that can be used in a workflow
component_t COMPONENTS[] = {

};
#define N_COMPONENTS 0

bool configure_workflow(uint32_t n_components,
        workflow_component_config_t *config,
        workflow_component_t **output_components) {

    // There will be the same number of components output as there are
    // configurations input; the only question will be how they are connected
    *output_components = spin1_malloc(
            sizeof(workflow_component_t) * n_components);
    if (!output_components) {
        log_error("Failed to allocate %u components", n_components);
        return false;
    }

    // Set up a list of pointers to keep the next components for later
    uint32_t *next_components[n_components];
    uint32_t n_outputs[n_components];
    uint32_t input_size[n_components];

    // First loop through all the components and initialize them and their
    // input pointers.
    workflow_component_config_t *next = config;
    for (uint32_t i = 0; i < n_components; i++) {

        // Find the component in the list
        if (next->component_id >= N_COMPONENTS) {
            log_error("Invalid component ID %u in configuration",
                    next->component_id);
            return false;
        }
        component_t *component = &COMPONENTS[next->component_id];
        workflow_component_t *output = &(*output_components)[i];

        // Set up the component with parameters (data follows the struct)
        void *params = &(next->next_components[next->n_inputs]);
        output->func = component->func;
        output->data = component->init(params);

        // Create the component input pointers
        output->n_inputs = next->n_inputs;
        output->input = spin1_malloc(sizeof(void *) * next->n_inputs);
        if (!output->input) {
            log_error("Failed to allocate %u input pointers for component %u",
                    next->n_inputs, next->component_id);
            return false;
        }

        // Create the component copy data flags
        output->copy_data = spin1_malloc(sizeof(bool) * next->n_inputs);
        if (!output->copy_data) {
            log_error("Failed to allocate %u copy data flags for component %u",
                    next->n_inputs, next->component_id);
            return false;
        }

        // Create the component output pointer
        output->output = spin1_malloc(sizeof(uint32_t) * next->output_size);
        if (!output->output) {
            log_error("Failed to allocate %u words of output data for "
                    "component %u",
                    next->output_size, next->component_id);
            return false;
        }

        // Keep track of data for later
        n_outputs[i] = next->n_outputs;
        next_components[i] = next->next_components;
        input_size[i] = next->input_size;

        next = &params[next->param_size];
    }

    // Keep track of how many inputs of the component have been used
    uint32_t next_input[n_components];

    // Now go through and set up the output pointers using the previous
    // component lists
    for (uint32_t i = 0; i < n_components; i++) {
        workflow_component_t *output = &(*output_components)[i];

        for (uint32_t j = 0; j < n_outputs[i]; j++) {
            uint32_t next_index = next_components[i][j];
            if (next_index >= n_components) {
                log_error("Invalid next component index %u in component %u",
                        next_components[i][j], i);
                return false;
            }
            workflow_component_t *next_component =
                    &(*output_components)[next_index];

            uint32_t next_i = next_input[i]++;
            if (next_i >= next_component->n_inputs) {
                log_error(
                        "Too many inputs for component %u, expected %u, got %u",
                        next_index, next_component->n_inputs, next_i);
                return false;
            }


            // If the component is a self-reference, copy the data first
            if (next_index == i) {
                output->copy_data[j] = true;
                uint32_t input_sz = input_size[i];
                next_component->input[next_i] = spin1_malloc(
                        sizeof(uint32_t) * input_sz);
            } else {
                output->copy_data[j] = false;

                // Set the input pointer to the output of the previous component
                next_component->input[next_i] = output->output;
            }
        }
    }

    return true;
}

void run_workflow(uint32_t n_components,
        workflow_component_t *components) {
    for (uint32_t i = 0; i < n_components; i++) {
        workflow_component_t *component = components[i];
        if (component && component->func) {
            component->func(component->data, component->n_inputs,
                    component->input,component->output);
        }
    }
}



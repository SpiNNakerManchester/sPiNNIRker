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
#include <debug.h>

//! A list of components that can be used in a workflow
static const component_t COMPONENTS[] = {

};
#define N_COMPONENTS 0

static bool init_workflow(workflow_config_t *config, workflow_t **workflow) {
    // Set up the workflow structure
    *workflow = spin1_malloc(sizeof(workflow_t));
    if (!*workflow) {
        log_error("Failed to allocate workflow structure");
        return false;
    }
    uint32_t n_components = config->n_components;
    (*workflow)->n_components = n_components;

    // There will be the same number of components output as there are
    // configurations input; the only question will be how they are connected
    workflow_component_t *components = spin1_malloc(
            sizeof(workflow_component_t) * n_components);
    if (!components) {
        log_error("Failed to allocate %u components", n_components);
        return false;
    }
    (*workflow)->components = components;

    (*workflow)->n_spike_inputs = config->n_spike_inputs;
    (*workflow)->n_sdram_inputs = config->n_sdram_inputs;
    (*workflow)->n_sdram_outputs = config->n_sdram_outputs;
    (*workflow)->running = false;
    (*workflow)->restart = false;
    (*workflow)->n_input_dmas_done = 0;
    (*workflow)->n_output_dmas_done = 0;
    (*workflow)->send_next_key = config->send_next_key;
    (*workflow)->wait_for_start_key = config->wait_for_start_key;
    (*workflow)->next_key = config->next_key;
    (*workflow)->start_key = config->start_key;
    (*workflow)->next_component = 0;

    return true;
}

static bool setup_components(uint32_t n_components,
        workflow_component_t *components,
        workflow_component_config_t **configs) {

    // First loop through all the components and initialize them and their
    // input pointers.
    workflow_component_config_t *next_component = (void *) configs[0];
    for (uint32_t i = 0; i < n_components; i++) {
        // Keep track of the configuration for this component, which is in
        // a variable position based on the parameters of the last component
        configs[i] = next_component;

        // Get the position of the parameters (after the configuration itself),
        // and move the next configuration pointer to the data after the
        // parameter data (won't be used if this is the last one)
        uint32_t *params = &(configs[i]->outputs[configs[i]->n_outputs]);
        next_component = (void *) &params[configs[i]->param_size];

        // Find the component in the list
        if (configs[i]->component_id >= N_COMPONENTS) {
            log_error("Invalid component ID %u in configuration",
                    configs[i]->component_id);
            return false;
        }

        // Set up the component with parameters (data follows the struct)
        const component_t *component = &COMPONENTS[configs[i]->component_id];
        components[i].func = component->func;
        components[i].data = component->init(params);


        // Create the component input pointers; note these don't have data
        // directly attached necessarily at this point, as they might just be
        // pointers to another component's output.  This will be set up later.
        uint32_t n_inputs = configs[i]->n_inputs;
        components[i].n_inputs = n_inputs;
        components[i].input = spin1_malloc(sizeof(void *) * n_inputs);
        if (!components[i].input) {
            log_error("Failed to allocate %u input pointers for component %u",
                    n_inputs, i);
            return false;
        }

        // Create the component copy data information
        components[i].copy_data_size = spin1_malloc(
                sizeof(uint32_t) * n_inputs);
        if (!components[i].copy_data_size) {
            log_error("Failed to allocate %u copy data size for component %u",
                    n_inputs, i);
            return false;
        }
        components[i].data_to_copy = spin1_malloc(sizeof(void*) * n_inputs);
        if (!components[i].data_to_copy) {
            log_error("Failed to allocate %u data to copy pointers for "
                    "component %u", n_inputs, i);
            return false;
        }

        // Ensure these values have been initialized to 0
        components[i].n_input_dmas_needed = 0;
        components[i].n_input_dmas_done = 0;

        // Create the component output pointer
        components[i].output = spin1_malloc(
                sizeof(uint32_t) * configs[i]->output_size);
        if (!components[i].output) {
            log_error("Failed to allocate %u words of output data for "
                    "component %u", configs[i]->output_size, i);
            return false;
        }
    }
    return true;
}

static bool setup_targets(uint32_t n_targets, uint32_t *target_indices,
        void *input_data, workflow_t *workflow,
        workflow_component_config_t **configs, uint32_t *next_inputs,
        uint32_t expected_input_size, bool is_sdram) {
    for (uint32_t j = 0; j < n_targets; j++) {
        uint32_t target_index = target_indices[j];
        if (target_index >= workflow->n_components) {
            log_error("Invalid target component index %u", target_index);
            return false;
        }

        // Use the next available input as the target for this component
        uint32_t max_index = workflow->components[target_index].n_inputs;
        uint32_t input_index = next_inputs[target_index]++;
        if (input_index >= max_index) {
            log_error("Too many inputs for component %u, expected %u, got %u",
                    target_index,  max_index, input_index);
            return false;
        }

        // Also need to create space for the component input
        workflow_component_t *component = &workflow->components[target_index];
        workflow_component_config_t *config = configs[target_index];
        if (expected_input_size > config->input_size) {
            log_error("Expected input size %u for component %u, got %u",
                    expected_input_size, target_index, config->input_size);
            return false;
        }
        component->input[input_index] = input_data;

        if (is_sdram) {
            component->n_input_dmas_needed++;
        }
    }
    return true;
}

static bool setup_spike_inputs(uint32_t n_spike_inputs,
        spike_input_config_t *spike_inputs, workflow_t *workflow,
        workflow_component_config_t **configs, uint32_t *next_inputs) {
    if (n_spike_inputs == 0) {
        workflow->spike_inputs = NULL;
        return true;
    }

    workflow->spike_inputs = spin1_malloc(
            sizeof(spike_input_t) * n_spike_inputs);
    if (!workflow->spike_inputs) {
        log_error("Failed to allocate %u spike inputs", n_spike_inputs);
        return false;
    }
    for (uint32_t i = 0; i < n_spike_inputs; i++) {
        spike_input_t *input = &workflow->spike_inputs[i];
        spike_input_config_t *config = &spike_inputs[i];
        input->key_info = config->key_info;

        // Make the spike list to be used by the inputs
        input->spikes = spin1_malloc(sizeof(spike_list_t)
                + (sizeof(spike_t) * config->max_spikes));
        // Check we have created a big enough buffer - +1 for the count
        uint32_t expected_size = input->max_spikes + 1;
        if (!setup_targets(config->n_targets, config->target_components,
                input->spikes, workflow, configs, next_inputs, expected_size,
                false)) {
            return false;
        }
        input->max_spikes = config->max_spikes;
        input->n_spikes_delay_lost = 0;
        input->n_spikes_overflow_lost = 0;
        input->n_spikes_received = 0;
    }
    return true;
}

static bool setup_sdram_inputs(uint32_t n_sdram_inputs,
        sdram_input_config_t *sdram_inputs, workflow_t *workflow,
        workflow_component_config_t **configs, uint32_t *next_inputs) {
    if (n_sdram_inputs == 0) {
        workflow->sdram_inputs = NULL;
        return true;
    }

    workflow->sdram_inputs = spin1_malloc(
            sizeof(sdram_input_t) * n_sdram_inputs);
    if (!workflow->sdram_inputs) {
        log_error("Failed to allocate %u SDRAM inputs", n_sdram_inputs);
        return false;
    }
    for (uint32_t i = 0; i < n_sdram_inputs; i++) {
        sdram_input_t *input = &workflow->sdram_inputs[i];
        sdram_input_config_t *config = &sdram_inputs[i];
        input->address = (void *) config->address;
        input->size_in_bytes = config->size_in_bytes;
        input->local_data = spin1_malloc(input->size_in_bytes);
        if (!input->local_data) {
            log_error("Failed to allocate %u bytes for SDRAM input %u",
                    input->size_in_bytes, i);
            return false;
        }

        input->n_targets = config->n_targets;
        uint32_t target_bytes = sizeof(uint32_t) * input->n_targets;
        input->target_components = spin1_malloc(target_bytes);
        if (!input->target_components) {
            log_error(
                    "Failed to allocate %u target components "
                    "for SDRAM input %u", config->n_targets, i);
            return false;
        }
        spin1_memcpy(input->target_components, config->target_components,
                target_bytes);

        if (!setup_targets(config->n_targets, config->target_components,
                input->local_data, workflow, configs, next_inputs, 0, true)) {
            return false;
        }
    }
    return true;
}

static bool setup_sdram_outputs(uint32_t n_sdram_outputs,
        sdram_output_config_t *sdram_outputs, workflow_t *workflow) {
    if (n_sdram_outputs == 0) {
        workflow->sdram_outputs = NULL;
        return true;
    }
    workflow->sdram_outputs = spin1_malloc(
            sizeof(sdram_output_t) * n_sdram_outputs);
    if (!workflow->sdram_outputs) {
        log_error("Failed to allocate %u SDRAM outputs", n_sdram_outputs);
        return false;
    }

    for (uint32_t i = 0; i < n_sdram_outputs; i++) {
        sdram_output_t *output = &workflow->sdram_outputs[i];
        sdram_output_config_t *config = &sdram_outputs[i];
        output->address = (void *) config->address;
        output->size_in_bytes = config->size_in_bytes;
        output->component_index = config->component_index;
    }
    return true;
}

static bool setup_outputs(uint32_t n_components, workflow_t *workflow,
        workflow_component_config_t **configs, uint32_t *next_inputs) {
    for (uint32_t i = 0; i < n_components; i++) {
        workflow_component_t *component = &workflow->components[i];
        workflow_component_config_t *config = configs[i];
        uint32_t n_outputs = config->n_outputs;
        uint32_t *outputs = config->outputs;

        for (uint32_t j = 0; j < n_outputs; j++) {
            uint32_t next_index = outputs[j];
            if (next_index >= n_components) {
                log_error("Invalid next component index %u in component %u",
                        next_index, i);
                return false;
            }
            workflow_component_t *target = &workflow->components[next_index];

            uint32_t input = next_inputs[next_index]++;
            if (input >= target->n_inputs) {
                log_error(
                        "Too many inputs for component %u, expected %u, got %u",
                        next_index, target->n_inputs, input);
                return false;
            }


            if (next_index == i) {
                // If the component is a self-reference, copy the data first
                uint32_t input_sz = config->input_size * sizeof(uint32_t);
                target->copy_data_size[input] = input_sz;
                target->data_to_copy[input] = component->output;
                target->input[input] = spin1_malloc(input_sz);
                if (!target->input[input]) {
                    log_error("Failed to allocate %u bytes for self-reference "
                            "input of component %u", input_sz, next_index);
                    return false;
                }
            } else {
                // If the component is not a self-reference, just set the
                // input pointer to the output of the previous component
                target->copy_data_size[input] = 0;
                target->data_to_copy[input] = NULL;
                target->input[input] = component->output;
            }
        }
    }
    return true;
}

static inline uint32_t get_core_id(uint32_t spike, key_info_t k_info) {
    return ((spike >> k_info.mask_shift) & k_info.core_mask);
}

static inline uint32_t get_local_id(uint32_t spike, key_info_t k_info) {
    uint32_t l_mask = ~(k_info.mask | (k_info.core_mask << k_info.mask_shift));
    uint32_t local = spike & l_mask;
    return local >> k_info.n_colour_bits;
}

static inline uint32_t get_colour(uint32_t spike, key_info_t k_info) {
    return spike & ((1 << k_info.n_colour_bits) - 1);
}

static bool spike_matches(uint32_t time, uint32_t key, spike_input_t *input,
        spike_t *spike) {
    if ((key & input->key_info.mask) == input->key_info.key) {
        input->n_spikes_received++;
        spike->delay = get_colour(time, input->key_info) -
                get_colour(key, input->key_info);
        if (spike->delay > get_colour(0xFFFFFFFF, input->key_info)) {
            // The spike delay is too large, so ignore it
            input->n_spikes_delay_lost++;
            return false;
        }
        spike->global_source_id = (get_core_id(key, input->key_info)
                * input->key_info.n_per_core)
                + get_local_id(key, input->key_info);
        return true;
    }
    return false;
}

static void reset_spikes(workflow_t *workflow) {
    for (uint32_t i = 0; i < workflow->n_spike_inputs; i++) {
        spike_input_t *input = &workflow->spike_inputs[i];
        input->spikes->n_spikes = 0;
    }
}

static void finish_workflow(workflow_t *workflow) {
    reset_spikes(workflow);
    if (workflow->send_next_key) {
        spin1_send_mc_packet(workflow->next_key, 0, NO_PAYLOAD);
    }
    workflow->next_component = 0;

    // If we have a flag to restart the workflow, start it now
    if (workflow->restart) {
        workflow->restart = false;
        workflow->running = true;
        spin1_trigger_user_event(0, 0);
    } else {
        workflow->running = false;
    }
}

//------ External API ------

bool configure_workflow(workflow_config_t *config, workflow_t **workflow) {
    uint32_t n_components = config->n_components;
    if (!init_workflow(config, workflow)) {
        return false;
    }

    // Get Pointers to the elements following the configuration structure
    spike_input_config_t *spike_inputs = (void *) &config->data[0];
    spike_input_config_t *last_spike_input =
            &spike_inputs[config->n_spike_inputs - 1];
    sdram_input_config_t *sdram_inputs =
            (void*) &last_spike_input->target_components[
                    last_spike_input->n_targets];
    sdram_input_config_t *last_sdram_input =
            &sdram_inputs[config->n_sdram_inputs - 1];
    sdram_output_config_t *sdram_outputs =
            (void*) &last_sdram_input->target_components[
                    last_sdram_input->n_targets];
    workflow_component_config_t *component_config =
            (void*) &sdram_outputs[config->n_sdram_outputs];

    // Set up a list of pointers to keep the next components for later
    workflow_component_config_t *configs[n_components];
    configs[0] = component_config;

    if (!setup_components((*workflow)->n_components, (*workflow)->components,
            configs)) {
        return false;
    }

    // Keep track of how many inputs of the component have been used
    uint32_t next_input[n_components];
    for (uint32_t i = 0; i < n_components; i++) {
        next_input[i] = 0;
    }

    // Setup the external inputs
    if (!setup_spike_inputs(config->n_spike_inputs, spike_inputs, *workflow,
            configs, next_input)) {
        return false;
    }
    if (!setup_sdram_inputs(config->n_sdram_inputs, sdram_inputs, *workflow,
            configs, next_input)) {
        return false;
    }
    if (!setup_sdram_outputs(config->n_sdram_outputs, sdram_outputs,
            *workflow)) {
        return false;
    }

    // Now go through and set up the output pointers using the previous
    // component lists
    if (!setup_outputs(n_components, *workflow, configs, next_input)) {
        return false;
    }

    return true;
}

bool process_packet(uint32_t time, uint32_t key, workflow_t *workflow) {
    for (uint32_t i = 0; workflow->n_spike_inputs; i++) {
        spike_input_t *input = &workflow->spike_inputs[i];
        spike_t spike;
        if (spike_matches(time, key, input, &spike)) {
            if (input->spikes->n_spikes >= input->max_spikes) {
                // If the spike input is full, ignore the spike
                input->n_spikes_overflow_lost++;
            } else {
                input->spikes->spikes[input->spikes->n_spikes++] = spike;
            }
            return true;
        }
    }
    return false;
}

static void setup_write_dma(workflow_t *workflow) {
    // Start the DMA for the next output - note tag is 0 for outputs since we
    // only need to know it is an output, not where it came from
    uint32_t next_output = workflow->n_output_dmas_done;
    if (next_output >= workflow->n_sdram_outputs) {
        log_error("No more SDRAM outputs to write to");
        return;
    }
    sdram_output_t *output = &workflow->sdram_outputs[next_output];
    workflow_component_t *component =
            &workflow->components[output->component_index];
    spin1_dma_transfer(0, output->address, component->output, DMA_WRITE,
            output->size_in_bytes);
}

void setup_read_dma(workflow_t *workflow) {
    // Start the DMA for the next input
    uint32_t next_input = workflow->n_input_dmas_done;
    if (next_input >= workflow->n_sdram_inputs) {
        log_error("No more SDRAM inputs to read from");
        return;
    }
    sdram_input_t *input = &workflow->sdram_inputs[next_input];

    dma_id_t id;
    id.is_input = 1;
    id.index = next_input;

    // Set up the DMA transfer for this input
    spin1_dma_transfer(id.id, input->address, input->local_data, DMA_READ,
            input->size_in_bytes);
}

void process_dma_complete(dma_id_t id, workflow_t *workflow) {
    // Turn off interrupts while we process this DMA
    uint32_t cpsr = spin1_int_disable();

    if (!id.is_input) {
        // If this is an output...
        // Increment the number of output DMAs done
        workflow->n_output_dmas_done++;

        if (workflow->n_output_dmas_done >= workflow->n_sdram_outputs) {
            // If we have done them all, finish the workflow
            workflow->n_output_dmas_done = 0;
            finish_workflow(workflow);
        } else {
            // If there are more DMAs to do, set the next one up
            setup_write_dma(workflow);
        }
    } else {
        workflow->n_input_dmas_done++;

        if (workflow->n_input_dmas_done < workflow->n_sdram_inputs) {
            // If there are more DMAs to do, set the next one up
            setup_read_dma(workflow);
        } else {
            // If all DMAs are done this round, we can reset them to be
            // started again the next run around.
            workflow->n_input_dmas_done = 0;
        }

        // Update the component input DMA counters
        bool start_run = false;
        sdram_input_t *input = &workflow->sdram_inputs[id.index];
        for (uint32_t i = 0; i < input->n_targets; i++) {
            uint32_t target = input->target_components[i];
            workflow_component_t *component = &workflow->components[target];
            component->n_input_dmas_done++;
            if (component->n_input_dmas_done >= component->n_input_dmas_needed
                    && !workflow->running
                    && workflow->next_component == target) {
                // If the next component has all its input DMAs done, and the
                // workflow is not currently running, start it again
                start_run = true;
            }
        }
        if (start_run) {
            // If we are ready to run the next component, run the workflow
            workflow->running = true;
            spin1_trigger_user_event(0, 0);
        }

    }

    // Re-enable interrupts
    spin1_mode_restore(cpsr);
}

void run_workflow(workflow_t *workflow) {
    // We assume that we are ready to start the component straight away
    workflow_component_t *component =
            &(workflow->components[workflow->next_component]);
    component->n_input_dmas_done = 0;

    // Copy any inputs that we need to copy (in case of self-loops)
    for (uint32_t j = 0; j < component->n_inputs; j++) {
        if (component->copy_data_size[j]) {
            spin1_memcpy(component->input[j],
                    component->data_to_copy[j],
                    component->copy_data_size[j]);
        }
    }

    // Call the component
    component->func(component->data, component->n_inputs,
            component->input, component->output);

    // If this is the last component, start the output DMA if needed
    if (workflow->next_component == workflow->n_components - 1) {
        uint32_t cpsr = spin1_int_disable();
        if (workflow->n_sdram_outputs > 0) {
            // Start the DMA if there are outputs to transfer
            setup_write_dma(workflow);
        } else {
            // Otherwise do what the DMA finish would have done
            finish_workflow(workflow);
        }

        // In any case, we can now stop
        spin1_mode_restore(cpsr);
        return;
    }

    // If the inputs for the next component are ready, start the next,
    // otherwise it will be started by the DMA completion.
    uint32_t cpsr = spin1_int_disable();
    workflow->next_component++;
    workflow_component_t *next_component =
            &workflow->components[workflow->next_component];
    if (next_component->n_input_dmas_done
            >= next_component->n_input_dmas_needed) {
        spin1_trigger_user_event(0, 0);
    } else {
        // If the inputs are not ready, pause running
        workflow->running = false;
    }
    spin1_mode_restore(cpsr);
}

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
//! \file conv1d_matrix.c
//! \brief Linear NIR component implementation (matrix inputs)
//! This is a 2D matrix multiplication of inputs by weights.
#include "conv1d_common.h"

typedef struct {
    // The main data commonly needed
    conv1d_config_t conv1d_config;
    // 1 / (input_channels * input_size) to find group
    div_const grp_div;
    // 1 / (input_size) to find input index
    div_const input_size_inv;
} conv1d_spike_config_t;

typedef struct {
    // The main data commonly needed
    conv1d_data_t *conv1d_data;
    // 1 / (input_channels * input_size) to find group
    div_const grp_div;
    // 1 / (input_size) to find input index
    div_const input_size_inv;
} conv1d_spike_data_t;

static void *conv1d_spikes_init(UNUSED uint32_t index, void *params) {
    conv1d_spike_data_t *data = spin1_malloc(sizeof(conv1d_spike_data_t));
    if (!data) {
        log_error("Failed to allocate conv1d_spike data structure");
        return (void *)0;
    }
    conv1d_spike_config_t *config = (conv1d_spike_config_t *)params;
    data->grp_div = config->grp_div;
    data->input_size_inv = config->input_size_inv;
    data->conv1d_data = conv1d_data_init(index, &config->conv1d_config);
    if (!data->conv1d_data) {
        log_error("Failed to initialize conv1d_data structure");
        return (void *)0;
    }

    return data;
}

static void conv1d_spikes_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    conv1d_spike_data_t *spike_data = data;
    conv1d_data_t *conv1d_data = spike_data->conv1d_data;

    // Get the output data
    int32_t *out_data = (int32_t *)output.data;

    // If there is a bias, we need to first set all the outputs to the bias
    if (conv1d_data->bias != 0) {
        for (uint32_t och = 0; och < conv1d_data->output_channels; och++) {
            for (uint32_t grp = 0; grp < conv1d_data->groups; grp++) {
                // Work out where we are in the output array for this group
                uint32_t o_idx = (grp * conv1d_data->output_channels + och)
                        * conv1d_data->output_size;
                for (uint32_t o = 0; o < conv1d_data->output_size; o++) {
                    out_data[o_idx + o] = conv1d_data->bias;
                }
            }
        }
    }

    // We now need to go through the spiking inputs
    for (uint32_t inp = 0; inp < n_inputs; inp++) {
        // Get the input data
        spike_list_t *in_data = (spike_list_t *)input[inp].data;

        // Go through the list of spikes and work out which input channel
        // and which group they are in
        for (uint32_t i = 0; i < in_data->n_spikes; i++) {
            spike_t spike = in_data->spikes[i];

            // Get the input channel and group from the source ID
            uint32_t global_source_id = spike.global_source_id;
            uint32_t grp = div_by_const(global_source_id, spike_data->grp_div);
            uint32_t rem_global_id = global_source_id -
                (grp * (conv1d_data->input_channels * conv1d_data->input_size));
            uint32_t ich = div_by_const(rem_global_id, spike_data->input_size_inv);

            // Work out where we are in the input array for this group
            uint32_t i_idx = rem_global_id - (ich * conv1d_data->input_size);

            // For each weight element from start to end
            for (int32_t w = -conv1d_data->half_conv_size;
                    w <= conv1d_data->half_conv_size; w++) {
                uint32_t out_i;
                // See if this input and weight maps properly to an output
                if (is_output(i_idx, w, conv1d_data, &out_i)) {

                    // For each output channel
                    for (uint32_t och = 0; och < conv1d_data->output_channels; och++) {

                        // Work out where we are in the output array for this group
                        uint32_t o_idx = (grp * conv1d_data->output_channels + och)
                                * conv1d_data->output_size;

                        // Work out where we are in the weights array for this group
                        uint32_t w_idx = ((grp * conv1d_data->output_channels + och)
                                * conv1d_data->input_channels + ich)
                                * (conv1d_data->half_conv_size * 2 + 1);

                        // Add the input multiplied by the weight to the output
                        out_data[o_idx + out_i] +=
                            conv1d_data->weights[w_idx + w];
                    }
                }
            }
        }
    }
}

const component_t conv1d_spikes = {
    .init = conv1d_spikes_init,
    .func = conv1d_spikes_exec,
    .dma_complete = NULL
};

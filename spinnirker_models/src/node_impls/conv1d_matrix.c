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

static inline void run_loop(uint32_t i_idx, uint32_t o_idx, uint32_t w_idx,
        conv1d_data_t *conv1d_data, uint32_t n_inputs, data_t *input,
        int32_t *out_data) {

    // If there is a bias value, we need to set all outputs to that first
    if (conv1d_data->bias != 0) {
        for (uint32_t o = 0; o < conv1d_data->output_size; o++) {
            out_data[o_idx + o] = conv1d_data->bias;
        }
    }

    // For each input element
    for (uint32_t input_i = 0; input_i < conv1d_data->input_size; input_i++) {

        // Keep the sum of the inputs so we only have to do it once
        int32_t input_sum = 0;
        uint32_t input_summed = 0;

        // For each weight element from start to end
        for (int32_t w = -conv1d_data->half_conv_size;
                w <= conv1d_data->half_conv_size; w++) {
            uint32_t out_i;
            // See if this input and weight maps properly to an output
            if (is_conv_1d_output(input_i, w, conv1d_data, &out_i)) {
                if (!input_summed) {
                    // If we haven't summed the inputs yet, do so now
                    for (uint32_t inp = 0; inp < n_inputs; inp++) {
                        int32_t *in_data = (int32_t *)input[inp].data;
                        input_sum += in_data[i_idx + input_i];
                    }
                    input_summed = 1;
                }

                // Add the summed input multiplied by the weight
                // to the output
                out_data[o_idx + out_i] += input_sum *
                    conv1d_data->weights[
                        w_idx + w + conv1d_data->half_conv_size];
            }
        }
    }
}

static void conv1d_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    conv1d_data_t *conv1d_data = data;

    // Get the output data
    int32_t *out_data = (int32_t *)output.data;

    // For each output channel
    for (uint32_t och = 0; och < conv1d_data->output_channels; och++) {
        // For each input channel
        for (uint32_t ich = 0; ich < conv1d_data->input_channels; ich++) {
            // For each group
            for (uint32_t grp = 0; grp < conv1d_data->groups; grp++) {
                // Work out where we are in the output array for this group
                uint32_t o_idx = (grp * conv1d_data->output_channels + och)
                        * conv1d_data->output_size;

                // Work out where we are in the input array for this group
                uint32_t i_idx = (grp * conv1d_data->input_channels + ich)
                        * conv1d_data->input_size;

                // Work out where we are in the weights array for this group
                uint32_t w_idx = ((grp * conv1d_data->output_channels + och)
                        * conv1d_data->input_channels + ich)
                        * (conv1d_data->half_conv_size * 2 + 1);

                run_loop(i_idx, o_idx, w_idx, conv1d_data, n_inputs, input,
                        out_data);
            }
        }
    }
}

const component_t conv1d_matrix = {
    .init = conv1d_data_init,
    .func = conv1d_matrix_exec,
    .dma_complete = NULL
};

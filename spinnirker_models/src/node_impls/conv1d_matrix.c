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

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "conv1d_matrix.h"

typedef struct {
    // The number of channels of the input
    uint32_t input_channels;
    // The number of channels of the output
    uint32_t output_channels;
    // The size of each group of the input
    uint32_t input_size;
    // The size of each group of the output
    uint32_t output_size;
    // Half the size of each group of the weights (-1)
    int32_t half_conv_size;
    // The stride to move the weights over the input
    uint32_t stride;
    // 1 / the stride to move the weights over the input (s1615)
    uint32_t inv_stride;
    // The padding to add to the start and end of the input
    uint32_t padding;
    // The dilation to apply to the weights
    uint32_t dilation;
    // The number of groups
    uint32_t groups;
    // The bias
    uint32_t bias;
    // The weights, as a kernel of (half_conv_size * 2 + 1) for each
    // combination of input and output channels and groups
    uint32_t weights[];
} conv1d_matrix_config_t;

typedef struct {
    // The number of channels of the input
    uint32_t input_channels;
    // The number of channels of the output
    uint32_t output_channels;
    // The size of each group of the input
    uint32_t input_size;
    // The size of each group of the output
    uint32_t output_size;
    // Half the size of each group of the weights (-1)
    int32_t half_conv_size;
    // The stride to move the weights over the input
    uint32_t stride;
    // 1 / the stride to move the weights over the input (s1615)
    uint32_t inv_stride;
    // The padding to add to the start and end of the input
    uint32_t padding;
    // The dilation to apply to the weights
    uint32_t dilation;
    // The number of groups
    uint32_t groups;
    // The bias
    uint32_t bias;
    // The weights
    uint32_t *weights;
} conv1d_matrix_data_t;

static void *conv1d_matrix_init(UNUSED uint32_t index, void *params) {
    conv1d_matrix_data_t *data = spin1_malloc(sizeof(conv1d_matrix_data_t));
    if (!data) {
        log_error("Failed to allocate linear matrix data structure");
        return (void *)0;
    }
    conv1d_matrix_config_t *config = (conv1d_matrix_config_t *)params;
    data->input_channels = config->input_channels;
    data->output_channels = config->output_channels;
    data->input_size = config->input_size;
    data->output_size = config->output_size;
    data->half_conv_size = config->half_conv_size;
    data->stride = config->stride;
    data->inv_stride = config->inv_stride;
    data->padding = config->padding;
    data->dilation = config->dilation;
    data->groups = config->groups;
    data->bias = config->bias;
    uint32_t weights_size = (data->half_conv_size * 2 + 1) * data->groups *
        data->input_channels * data->output_channels;
    data->weights = spin1_malloc(weights_size * sizeof(uint32_t));
    if (!data->weights) {
        log_error("Failed to allocate %u weights for conv1d matrix", weights_size);
        return (void *)0;
    }
    spin1_memcpy(data->weights, config->weights, weights_size * sizeof(uint32_t));
    return data;
}

static uint32_t is_output(uint32_t input_i, int32_t kernel_i,
        conv1d_matrix_data_t *conv1d_data, uint32_t *output) {
    int32_t dividend =
        (input_i - (conv1d_data->half_conv_size * conv1d_data->dilation)
                        - conv1d_data->padding)
            - (kernel_i * conv1d_data->dilation);
    // Scale by stride reciprocal to get an S1615 value
    int32_t value = __I32((__I64(dividend) * __I64(conv1d_data->inv_stride)) >> 15);

    // If the value is a positive integer, return true
    if (value >= 0 && (value & 0x7FFF) == 0) {
        *output = (uint32_t) (value >> 15);
        return true;
    }
    return false;
}

static void conv1d_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    conv1d_matrix_data_t *conv1d_data = data;

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
                        if (is_output(input_i, w, conv1d_data, &out_i)) {
                            if (!input_summed) {
                                // If we haven't summed the inputs yet, do so now
                                for (uint32_t inp = 0; inp < n_inputs; inp++) {
                                    int32_t *in_data = (int32_t *)input[inp].data;
                                    input_sum += in_data[i_idx + input_i];
                                }
                                input_summed = 1;
                            }

                            // Add the summed input multiplied by the weight to the output
                            out_data[o_idx + out_i] += input_sum *
                                conv1d_data->weights[w_idx + w];
                        }
                    }
                }
            }
        }
    }
}

const component_t conv1d_matrix = {
    .init = conv1d_matrix_init,
    .func = conv1d_matrix_exec,
    .dma_complete = NULL
};

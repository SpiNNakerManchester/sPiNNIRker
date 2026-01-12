/*
 Copyright (c) 2026 The University of Manchester
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
//! \file conv1d_common.c
//! \brief Convolution 1D common functions

#include "conv1d_common.h"

conv1d_data_t *conv1d_data_init(UNUSED uint32_t index, void *params) {
    conv1d_data_t *data = spin1_malloc(sizeof(conv1d_data_t));
    if (!data) {
        log_error("Failed to allocate conv1d_data structure");
        return (void *)0;
    }
    conv1d_config_t *config = (conv1d_config_t *)params;
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

uint32_t is_output(uint32_t input_i, int32_t kernel_i,
        conv1d_data_t *conv1d_data, uint32_t *output) {
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

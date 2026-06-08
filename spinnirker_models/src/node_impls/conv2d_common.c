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

#include "conv2d_common.h"

void *conv2d_data_init(UNUSED uint32_t index, void *params) {
    conv2d_data_t *data = spin1_malloc(sizeof(conv2d_data_t));
    if (!data) {
        log_error("Failed to allocate conv1d_data structure");
        return (void *)0;
    }
    conv2d_config_t *config = (conv2d_config_t *)params;
    data->input_channels = config->input_channels;
    data->output_channels = config->output_channels;
    data->input_width = config->input_width;
    data->input_height = config->input_height;
    data->output_width = config->output_width;
    data->output_height = config->output_height;
    data->half_conv_width = config->half_conv_width;
    data->half_conv_height = config->half_conv_height;
    data->stride_x = config->stride_x;
    data->stride_y = config->stride_y;
    data->inv_stride_x = config->inv_stride_x;
    data->inv_stride_y = config->inv_stride_y;
    data->padding_x = config->padding_x;
    data->padding_y = config->padding_y;
    data->dilation_x = config->dilation_x;
    data->dilation_y = config->dilation_y;
    data->groups = config->groups;
    data->bias = config->bias;
    uint32_t weights_size = (data->half_conv_width * 2 + 1)
            * (data->half_conv_height * 2 + 1) * data->groups
            * data->input_channels * data->output_channels;
    data->weights = spin1_malloc(weights_size * sizeof(uint32_t));
    if (!data->weights) {
        log_error("Failed to allocate %u weights for conv1d matrix", weights_size);
        return (void *)0;
    }
    spin1_memcpy(data->weights, config->weights, weights_size * sizeof(uint32_t));
    return data;
}

static inline uint32_t is_output_dim(uint32_t input_i, int32_t kernel_i,
        int32_t half_conv_size, uint32_t dilation, uint32_t padding,
        uint32_t inv_stride, uint32_t *output) {
    int32_t dividend =
        (input_i - (half_conv_size * dilation) - padding)
        - (kernel_i * dilation);
    // Scale by stride reciprocal to get an S1615 value
    int32_t value = __I32((__I64(dividend) * __I64(inv_stride)) >> 15);

    // If the value is a positive integer, return true
    if (value >= 0 && (value & 0x7FFF) == 0) {
        *output = (uint32_t) (value >> 15);
        return true;
    }
    return false;
}

uint32_t is_conv_2d_output(uint32_t input_i_x, uint32_t input_i_y,
        int32_t kernel_i_x, int32_t kernel_i_y,
        conv2d_data_t *conv2d_data, uint32_t *output_x, uint32_t *output_y) {
    if (!is_output_dim(input_i_x, kernel_i_x, conv2d_data->half_conv_width,
            conv2d_data->dilation_x, conv2d_data->padding_x,
            conv2d_data->inv_stride_x, output_x)) {
        return false;
    }
    if (!is_output_dim(input_i_y, kernel_i_y, conv2d_data->half_conv_height,
            conv2d_data->dilation_y, conv2d_data->padding_y,
            conv2d_data->inv_stride_y, output_y)) {
        return false;
    }
    return true;
}

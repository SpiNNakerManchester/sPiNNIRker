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
//! \file conv1d_common.h
//! \brief Convolution 1D common header file
#ifndef CONV1D_COMMON_H
#define CONV1D_COMMON_H

#include <arm_acle.h>
#include <stdfix-full-iso.h>
#include <spin1_api.h>
#include <debug.h>
#include "../component.h"

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
    uint32_t half_conv_size;
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
} conv1d_config_t;

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
} conv1d_data_t;

extern void *conv1d_data_init(UNUSED uint32_t index, void *params);

extern uint32_t is_output(uint32_t input_i, int32_t kernel_i,
        conv1d_data_t *conv1d_data, uint32_t *output);

#endif // CONV1D_COMMON_H

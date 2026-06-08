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
//! \file conv2d_common.h
//! \brief Convolution 2D common header file
#ifndef CONV2D_COMMON_H
#define CONV2D_COMMON_H

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
    // The width of each group of the input
    uint32_t input_width;
    // The height of each group of the input
    uint32_t input_height;
    // The width of each group of the output
    uint32_t output_width;
    // The height of each group of the output
    uint32_t output_height;
    // Half the width of each group of the weights (-1)
    int32_t half_conv_width;
    // Half the height of each group of the weights (-1)
    int32_t half_conv_height;
    // The stride to move the weights over the input in the x direction
    uint32_t stride_x;
    // The stride to move the weights over the input in the y direction
    uint32_t stride_y;
    // 1 / the stride to move the weights over the input (s1615) in the x
    // direction
    uint32_t inv_stride_x;
    // 1 / the stride to move the weights over the input (s1615) in the y
    // direction
    uint32_t inv_stride_y;
    // The padding to add to the start and end of the input in the x direction
    uint32_t padding_x;
    // The padding to add to the start and end of the input in the y direction
    uint32_t padding_y;
    // The dilation to apply to the weights in the x direction
    uint32_t dilation_x;
    // The dilation to apply to the weights in the y direction
    uint32_t dilation_y;
    // The number of groups
    uint32_t groups;
    // The bias
    uint32_t bias;
    // The weights, as a kernel of
    // (half_conv_width * 2 + 1 by half_conv_height * 2 + 1)
    // for each combination of input and output channels and groups
    uint32_t weights[];
} conv2d_config_t;

typedef struct {
    // The number of channels of the input
    uint32_t input_channels;
    // The number of channels of the output
    uint32_t output_channels;
    // The width of each group of the input
    uint32_t input_width;
    // The height of each group of the input
    uint32_t input_height;
    // The width of each group of the output
    uint32_t output_width;
    // The height of each group of the output
    uint32_t output_height;
    // Half the width of each group of the weights (-1)
    int32_t half_conv_width;
    // Half the height of each group of the weights (-1)
    int32_t half_conv_height;
    // The stride to move the weights over the input in the x direction
    uint32_t stride_x;
    // The stride to move the weights over the input in the y direction
    uint32_t stride_y;
    // 1 / the stride to move the weights over the input (s1615) in the x
    // direction
    uint32_t inv_stride_x;
    // 1 / the stride to move the weights over the input (s1615) in the y
    // direction
    uint32_t inv_stride_y;
    // The padding to add to the start and end of the input in the x direction
    uint32_t padding_x;
    // The padding to add to the start and end of the input in the y direction
    uint32_t padding_y;
    // The dilation to apply to the weights in the x direction
    uint32_t dilation_x;
    // The dilation to apply to the weights in the y direction
    uint32_t dilation_y;
    // The number of groups
    uint32_t groups;
    // The bias
    uint32_t bias;
    // The weights
    uint32_t *weights;
} conv2d_data_t;

extern void *conv2d_data_init(UNUSED uint32_t index, void *params);

extern uint32_t is_conv_2d_output(uint32_t input_i_x, uint32_t input_i_y,
        int32_t kernel_i_x, int32_t kernel_i_y,
        conv2d_data_t *conv2d_data, uint32_t *output_x, uint32_t *output_y);

#endif // CONV2D_COMMON_H

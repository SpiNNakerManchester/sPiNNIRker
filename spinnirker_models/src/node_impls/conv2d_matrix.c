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

#include "conv2d_common.h"

static inline void run_loop(uint32_t i_idx, uint32_t o_idx, uint32_t w_idx,
        conv2d_data_t *conv2d_data, uint32_t n_inputs, data_t *input,
        int32_t *out_data) {
    // If there is a bias value, we need to set all outputs to that first
    if (conv2d_data->bias != 0) {
        uint32_t output_size = conv2d_data->output_width
                * conv2d_data->output_height;
        for (uint32_t o = 0; o < output_size; o++) {
            out_data[o_idx + o] = conv2d_data->bias;
        }
    }

    uint32_t conv_width = conv2d_data->half_conv_width * 2 + 1;

    // For each input element
    for (uint32_t input_i_x = 0; input_i_x < conv2d_data->input_width; input_i_x++) {
        for (uint32_t input_i_y = 0; input_i_y < conv2d_data->input_height; input_i_y++) {

            // Keep the sum of the inputs so we only have to do it once
            int32_t input_sum = 0;
            uint32_t input_summed = 0;

            // For each weight element
            for (int32_t w_x = -conv2d_data->half_conv_width;
                    w_x <= conv2d_data->half_conv_width; w_x++) {
                for (int32_t w_y = -conv2d_data->half_conv_height;
                        w_y <= conv2d_data->half_conv_height; w_y++) {
                    uint32_t out_i_x;
                    uint32_t out_i_y;
                    // See if this input and weight maps properly to
                    // an output
                    if (is_conv_2d_output(input_i_x, input_i_y, w_x,
                            w_y, conv2d_data, &out_i_x, &out_i_y)) {
                        if (!input_summed) {
                            // If we haven't summed the inputs yet,
                            // do so now
                            for (uint32_t inp = 0; inp < n_inputs; inp++) {
                                int32_t *in_data = (int32_t *) input[inp].data;
                                uint32_t in_i = i_idx
                                        + (input_i_y * conv2d_data->input_width)
                                        + input_i_x;
                                input_sum += in_data[in_i];
                            }
                            input_summed = 1;
                        }

                        // Add the summed input multiplied by the weight to the output
                        uint32_t out_i = o_idx
                                + (out_i_y * conv2d_data->output_width)
                                + out_i_x;
                        uint32_t w_i =
                                w_idx
                                + ((w_y + conv2d_data->half_conv_height)
                                        * conv_width)
                                + (w_x + conv2d_data->half_conv_width);
                        out_data[out_i] += input_sum * conv2d_data->weights[w_i];
                    }
                }
            }
        }
    }
}

static void conv2d_matrix_exec(void *data, uint32_t n_inputs, data_t *input,
        data_t output) {
    // Get the data structure
    conv2d_data_t *conv2d_data = data;

    // Get the output data
    int32_t *out_data = (int32_t *)output.data;

    // For each output channel
    for (uint32_t och = 0; och < conv2d_data->output_channels; och++) {
        // For each input channel
        for (uint32_t ich = 0; ich < conv2d_data->input_channels; ich++) {
            // For each group
            for (uint32_t grp = 0; grp < conv2d_data->groups; grp++) {

                // Work out where we are in the output array for this group
                uint32_t o_idx = (grp * conv2d_data->output_channels + och)
                        * conv2d_data->output_width * conv2d_data->output_height;

                // Work out where we are in the input array for this group
                uint32_t i_idx = (grp * conv2d_data->input_channels + ich)
                        * conv2d_data->input_width * conv2d_data->input_height;

                // Work out where we are in the weights array for this group
                uint32_t w_idx = ((grp * conv2d_data->output_channels + och)
                        * conv2d_data->input_channels + ich)
                        * (conv2d_data->half_conv_width * 2 + 1)
                        * (conv2d_data->half_conv_height * 2 + 1);

                run_loop(i_idx, o_idx, w_idx, conv2d_data, n_inputs, input, out_data);
            }
        }
    }
}

const component_t conv2d_matrix = {
    .init = conv2d_data_init,
    .func = conv2d_matrix_exec,
    .dma_complete = NULL
};

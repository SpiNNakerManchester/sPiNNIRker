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
//! \file delay.c
//! \brief Delay NIR component implementation
#include <stdint.h>
#include <spin1_api.h>
#include <arm_acle.h>
#include <debug.h>
#include "delay.h"

typedef struct {
    //! The delay in time steps
    uint32_t delay;

    //! The size of the buffer to save each timestep, in words
    uint32_t buffer_size;
} delay_config_t;

typedef struct {
    //! The configuration of the delay
    delay_config_t config;

    dma_id_t dma_id;

    //! The current time step
    uint32_t time;

    //! The circular buffer to hold the delayed data
    uint32_t *sdram_buffer;

    //! The next buffer transferred from SDRAM for next timestep
    uint32_t *next_buffer;

    //! The next position in the buffer to read from for next timestep
    uint32_t read_pos;

    //! The next position in the buffer to write to for next timestep
    uint32_t write_pos;

    //! Whether the DMA is in progress
    bool dma_in_progress;
} delay_data_t;

static void* delay_init(uint32_t index, void *params) {
    // Cast the parameters to the delay configuration
    delay_config_t *config = (delay_config_t *) params;

    delay_data_t *delay_data = spin1_malloc(sizeof(delay_data_t));
    if (!delay_data) {
        log_error("Failed to allocate delay data structure");
        return (void *) 0;
    }

    delay_data->config = *config;
    delay_data->time = 0;
    uint32_t total_size = config->buffer_size * config->delay * sizeof(uint32_t);
    delay_data->sdram_buffer = sark_xalloc(sv->sdram_heap, total_size, 0,
            ALLOC_LOCK);
    if (!delay_data->sdram_buffer) {
        log_error("Failed to allocate delay buffer of size %u", total_size);
        return (void *) 0;
    }
    uint32_t one_size = config->buffer_size * sizeof(uint32_t);
    delay_data->next_buffer = spin1_malloc(one_size);
    if (!delay_data->next_buffer) {
        log_error("Failed to allocate next buffer of size %u", one_size);
        return (void *) 0;
    }
    delay_data->read_pos = 0;
    delay_data->write_pos = total_size - one_size;
    delay_data->dma_in_progress = false;
    delay_data->dma_id.index = index;
    delay_data->dma_id.is_component = 1;

    return delay_data;
}

static void delay_exec(void *data, uint32_t n_inputs, data_t *input,
        UNUSED data_t output) {
    // Get the data structure
    delay_data_t *delay_data = data;

    if (n_inputs != 1) {
        log_error("Delay component only supports one input");
        return;
    }

    // If a DMA is in progress, wait for it to complete
    uint32_t cpsr = spin1_int_disable();
    while (delay_data->dma_in_progress) {
        spin1_wfi();
    }
    spin1_mode_restore(cpsr);

    // Copy the input data to the SDRAM buffer at the write position
    spin1_memcpy(&(delay_data->sdram_buffer[delay_data->write_pos]),
            input[0].data,
            delay_data->config.buffer_size * sizeof(uint32_t));

    // If we are past the first delay, copy the data that has been delayed
    // from the DMA buffer to the output
    if (delay_data->time >= delay_data->config.delay) {
        spin1_memcpy(output.data,
                delay_data->next_buffer,
                delay_data->config.buffer_size * sizeof(uint32_t));
    }

    // Start the DMA to copy the next buffer from the correct position in the
    // circular buffer if there is data to copy
    if (delay_data->time + 1 >= delay_data->config.delay) {
        delay_data->dma_in_progress = 1;
        spin1_dma_transfer(delay_data->dma_id.id,
                &(delay_data->sdram_buffer[delay_data->read_pos]),
                delay_data->next_buffer, DMA_READ,
                delay_data->config.buffer_size * sizeof(uint32_t));
    }

    // Update the time and positions
    delay_data->time++;
    delay_data->read_pos += delay_data->config.buffer_size;
    if (delay_data->read_pos >= delay_data->config.buffer_size
            * delay_data->config.delay) {
        delay_data->read_pos = 0;
    }
    delay_data->write_pos += delay_data->config.buffer_size;
    if (delay_data->write_pos >= delay_data->config.buffer_size
            * delay_data->config.delay) {
        delay_data->write_pos = 0;
    }
}

void delay_dma_complete(UNUSED dma_id_t id, void *data) {
    // Get the data structure
    delay_data_t *delay_data = data;
    delay_data->dma_in_progress = 0;
}

const component_t delay_component = {
    .init = delay_init,
    .func = delay_exec,
    .dma_complete = delay_dma_complete
};

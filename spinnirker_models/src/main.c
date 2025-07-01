/*
 * Copyright (c) 2017 The University of Manchester
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * \dir
 * \brief Implementation of the synapse expander and delay expander
 * \file
 * \brief The synapse expander for neuron cores
 */
#include "workflow.h"

#include <spin1_api.h>
#include <bit_field.h>
#include <data_specification.h>
#include <simulation.h>
#include <recording.h>
#include <debug.h>

enum REGIONS {
    //! The system region
    SYSTEM_REGION = 0,
    //! The configuration region
    CONFIG_REGION,
    //! The workflow region
    WORKFLOW_REGION,
    //! The recording metadata region
    RECORDING_METADATA_REGION,
};

enum PRIORITY {
    // USER and SDP are background tasks
    USER = 1,
    SDP = 1,
    // Timer and DMA should interrupt the background tasks
    TIMER = 0,
    DMA = 0,
    // The Multicast callback should interrupt everything to get the data off
    // the network.
    MULTICAST = -1,
    // The simulation should not set up DMA as this is handled separately
    SIM_DMA = -2
};

typedef struct {
    //! The maximum number of input packets
    uint32_t max_input_packets;
} config_t;

typedef struct {
    uint32_t n_packets;
    uint32_t *packets;
} packet_buffer_t;

//! The timer period in microseconds
static uint32_t timer_period;

//! The number of simulation ticks
static uint32_t simulation_ticks;

//! Whether the simulation should run indefinitely
static uint32_t infinite_run;

//! The flags for recording regions
static uint32_t recording_flags;

//! The current simulation tick
static uint32_t time;

//! A configured workflow to be run
static workflow_t *workflow;

//! The main configuration
static config_t config;

//! Buffers of packets received but yet to be processed;
//! we keep 2 of them to swap out each timer tick
static packet_buffer_t input_packets[2];

//! The index of the buffer to put spikes in now
static uint32_t write_buffer = 0;

//! The index of the buffer to read spike from now
static uint32_t read_buffer = 0;

//! The number of packet buffer overflows
static uint32_t n_packet_overflows = 0;

/**
 * \brief Set up the workflow.
 */
static bool initialize(void) {
    // Get the addresses of the regions
    data_specification_metadata_t *ds_regions =
            data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(ds_regions)) {
        return false;
    }

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(
            data_specification_get_region(SYSTEM_REGION, ds_regions),
            APPLICATION_NAME_HASH, &timer_period, &simulation_ticks,
            &infinite_run, &time, SDP, SIM_DMA)) {
        return false;
    }

    if (!recording_initialize(
            data_specification_get_region(RECORDING_METADATA_REGION,
                    ds_regions),
            &recording_flags)) {
        return false;
    }

    // Set up the workflow components
    if (!configure_workflow(data_specification_get_region(WORKFLOW_REGION,
            ds_regions), &workflow)) {
        return false;
    }

    // Set up the input spike buffer
    config_t *sdram_config = data_specification_get_region(CONFIG_REGION,
            ds_regions);
    config = *sdram_config;
    for (uint32_t i = 0; i < 2; i++) {
        input_packets[i].n_packets = 0;
        input_packets[i].packets = spin1_malloc(sizeof(uint32_t) *
                config.max_input_packets);
        if (!input_packets[i].packets) {
            log_error("Failed to allocate packet buffer of %u packets",
                    config.max_input_packets);
            return false;
        }
    }
    read_buffer = 0;
    write_buffer = 0;
    return true;
}

static void resume_callback(void) {
    if (recording_flags) {
        recording_reset();
    }
}

static void start_workflow(void) {
    // If the workflow is currently running, just mark it to restart
    if (workflow->running) {
        workflow->restart = true;
        return;
    }

    // If the workflow has SDRAM inputs, start these running
    if (workflow->n_sdram_inputs > 0) {
        workflow->running = true;
        workflow->n_input_dmas_done = 0;
        workflow->n_output_dmas_done = 0;
        setup_read_dma(workflow);

        // If the first component doesn't need SDRAM inputs, start running
        if (workflow->components[0].n_input_dmas_needed == 0) {
            spin1_trigger_user_event(0, 0);
        }

        return;
    }

    // The workflow must be ready to run, so start it
    workflow->running = true;
    spin1_trigger_user_event(0, 0);
}

void timer_callback(uint time, uint unused0) {
    use(unused0);
    // Increment the time
    time++;

    // If we have reached the end of the simulation, stop
    if (simulation_is_finished()) {
        log_info("Simulation complete at time %u", time);
        simulation_handle_pause_resume(resume_callback);

        if (recording_flags) {
            recording_finalise();
        }

        simulation_ready_to_read();
        return;
    }

    // Switch over the packet buffers
    read_buffer = write_buffer;
    write_buffer = (write_buffer + 1) & 0x1;

    // If we need to wait for a packet to start the workflow, do nothing
    if (workflow->wait_for_start_key) {
        return;
    }
    start_workflow();
}

void multicast_packet_callback(uint key, uint payload) {
    use(key);
    use(payload);
    // If the workflow is waiting for a start key, check if this is it
    if (workflow->wait_for_start_key && key == workflow->start_key) {
        start_workflow();
        return;
    }

    // Store the spikes, or not if already full
    packet_buffer_t *packet_buffer = &input_packets[write_buffer];
    if (packet_buffer->n_packets >= config.max_input_packets) {
        n_packet_overflows++;
        return;
    }
    packet_buffer->packets[packet_buffer->n_packets++] = key;
}

void user_callback(uint unused0, uint unused1) {
    use(unused0);
    use(unused1);

    // If we are about to start component 0, first process any packets
    if (workflow->next_component == 0) {
        // Process the spikes in the current buffer
        packet_buffer_t *packet_buffer = &input_packets[read_buffer];
        for (uint32_t i = 0; i < packet_buffer->n_packets; i++) {
            process_packet(time, packet_buffer->packets[i], workflow);
        }
        packet_buffer->n_packets = 0;
    }

    run_workflow(workflow);
}

void dma_callback(uint unused0, uint tag) {
    use(unused0);
    dma_id_t id;
    id.id = tag;
    process_dma_complete(id, workflow);
}

//! Entry point
void c_main(void) {
    // Load DTCM data
    time = 0;
    if (!initialize()) {
        log_error("Error in initialisation - exiting!");
        rt_error(RTE_SWERR);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Set timer tick (in microseconds)
    spin1_set_timer_tick(timer_period);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);
    spin1_callback_on(MC_PACKET_RECEIVED, multicast_packet_callback, MULTICAST);
    spin1_callback_on(USER_EVENT, user_callback, USER);
    spin1_callback_on(DMA_TRANSFER_DONE, dma_callback, DMA);

    simulation_run();
}

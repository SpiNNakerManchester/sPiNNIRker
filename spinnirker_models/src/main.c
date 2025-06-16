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
#include <data_specification.h>
#include <simulation.h>
#include <recording.h>
#include <debug.h>

enum REGIONS {
    //! The system region
    SYSTEM_REGION = 0,
    //! The workflow region
    WORKFLOW_REGION,
    //! The recording metadata region
    RECORDING_METADATA_REGION,
};

enum PRIORITY {
    SDP = 1,
    TIMER = 1,
    DMA = 0,
    MULTICAST = -1,
};

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
            &infinite_run, &time, SDP, DMA)) {
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
    return true;
}

static void resume_callback(void) {
    if (recording_flags) {
        recording_reset();
    }
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

    // Run the workflow for this timer tick
    run_workflow(workflow);

    if (workflow->output_recorded) {
        // If record output, do recording
        if (workflow->output_spikes) {
            // TODO: Record spikes
        } else {
            // TODO: Record data
        }
    } else if (workflow->output_spikes) {
        // Otherwise if the output is spikes, we need to send spikes
        // TODO: Send spikes
    }
}

void multicast_packet_callback(uint key, uint payload) {
    use(key);
    use(payload);
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
    spin1_callback_on(
            MC_PACKET_RECEIVED, multicast_packet_callback, MULTICAST);

    simulation_run();
}

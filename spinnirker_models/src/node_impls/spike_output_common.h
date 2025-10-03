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
//! \file spike_output_common.h
//! \brief Common spike output functions
#include <spin1_api.h>
#include <debug.h>
#include "../component.h"

static void spike(spike_list_t *output, uint32_t key, uint32_t index) {
    // Send the spike first as we can always do this
    spin1_send_mc_packet(key + index, 0, 0);

    // Store the spike in the output list if there is space; there might
    // not be if the output is not being used by another component, so no
    // need to error.
    if (output->n_spikes >= output->max_spikes) {
        return;
    }
    output->spikes[output->n_spikes++].global_source_id = index;
}

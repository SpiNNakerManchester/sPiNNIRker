// DO NOT EDIT! THIS FILE WAS GENERATED FROM ../../src/workflow.h

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
//! \file workflow.h
//! \brief Workflow definitions

#include <stdbool.h>
#include <stdint.h>
#include <bit_field.h>

//! \brief Defines a function type for components in a workflow.
//! Note no input is the same as the output.
//! \param[in] data: Pointer to the workflow data structure that holds
//!                  the state and configuration for the workflow
//! @param[in] n_inputs: Number of input data pointers for the component
//! \param[in] input: Pointer to input data for the component
//! \param[out] output: Pointer to output data for the component
typedef void (*component_func)(void *data, uint32_t n_inputs, void **input,
        void *output);

//! \brief Defines a function type for initializing components in a workflow
//! \param[in] params: Pointer to parameters for the component
//! \return Pointer to the initialized component data structure
typedef void* (*component_init)(void *params);

//! \brief Defines a function type for deinitializing components in a workflow
//! \param[in] data: Pointer to the component data structure to be deinitialized
typedef void (*component_deinit)(void *data);

//! A component that could be used in a workflow.
typedef struct {
    //! Function to call for this component
    component_func func;
    //! Function to call to initialize this component
    component_init init;
    //! Function to call to deinitialize this component
    component_deinit deinit;
} component_t;

//! Information about the key and mask for a spike input
typedef struct {
    //! The key for the spike input
    uint32_t key;
    //! The mask for the spike input
    uint32_t mask;
    //! The number of bits of key used for colour (0 if no colour)
    uint32_t n_colour_bits: 3;
    //! The shift to apply to the key to get the core part
    uint32_t mask_shift: 13;
    //! The mask to apply to the key once shifted to get the core index
    uint32_t core_mask: 16;
    //! The number of atoms per core
    uint32_t n_per_core;
} key_info_t;

typedef struct {
    // The source ID of the spike
    uint32_t global_source_id: 28;
    // The delay of the spike in reaching this core from the source core
    uint32_t delay: 4;
} spike_t;

typedef struct {
    //! The number of spikes in the list
    uint32_t n_spikes;
    //! The spikes in the list
    spike_t spikes[];
} spike_list_t;

//! An input that receives spikes.
typedef struct {
    //! Information about the key and mask for the spike input
    key_info_t key_info;
    //! The maximum number of spikes to allow in the input
    uint32_t max_spikes;
    //! The location where the spikes are stored
    spike_list_t *spikes;
    //! The number of spikes lost to overflow
    uint32_t n_spikes_overflow_lost;
    //! The number of spikes lost to delay overflow
    uint32_t n_spikes_delay_lost;
    //! The number of spikes received
    uint32_t n_spikes_received;
} spike_input_t;

//! An input that receives data from SDRAM using DMA
typedef struct {
    //! The address in SDRAM to read from
    void *address;
    //! The size of the data to read, in bytes
    uint32_t size_in_bytes;
    //! The local memory allocated for the input data
    void *local_data;
    //! The number of components that the SDRAM input targets
    uint32_t n_targets;
    //! List of component IDs targeted
    uint32_t *target_components;
} sdram_input_t;

//! An output that sends data to SDRAM using DMA
typedef struct {
    //! The address in SDRAM to write to
    void *address;
    //! The size of the data to write, in bytes
    uint32_t size_in_bytes;
    //! The sources of the SDRAM output
    uint32_t component_index;
} sdram_output_t;

//! A component in a workflow
typedef struct {
    //! Function to call for this component
    component_func func;
    //! Pointer to the workflow data structure
    void *data;
    //! The number of inputs to this component
    uint32_t n_inputs;
    //! Input data pointer(s)
    void **input;
    //! For each input how much data to copy (0 for no copy)
    uint32_t *copy_data_size;
    //! Pointers to data to copy to the input
    void **data_to_copy;
    //! The number of DMAs needed before starting this component
    uint32_t n_input_dmas_needed;
    //! The number of DMAs done before starting this component
    uint32_t n_input_dmas_done;
    //! Output data pointer
    void *output;
} workflow_component_t;

typedef struct {
    //! The number of spike inputs into the workflow
    uint32_t n_spike_inputs;
    //! The spike inputs into the workflow
    spike_input_t *spike_inputs;
    //! The number of SDRAM inputs into the workflow
    uint32_t n_sdram_inputs;
    //! The SDRAM inputs into the workflow
    sdram_input_t *sdram_inputs;
    //! The number of SDRAM outputs from the workflow
    uint32_t n_sdram_outputs;
    //! The SDRAM outputs from the workflow
    sdram_output_t *sdram_outputs;
    //! The number of components in the workflow
    uint32_t n_components;
    //! The components in the workflow
    workflow_component_t *components;
    //! The number of input DMA operations done
    uint32_t n_input_dmas_done;
    //! The number of output DMA operations done
    uint32_t n_output_dmas_done;
    //! The next component to be run
    uint32_t next_component;
    //! Whether the workflow is currently running
    bool running;
    //! Whether the workflow should be restarted on completion
    bool restart;
    //! Whether to send the next key on completion
    bool send_next_key;
    //! Whether to wait for the start key to start
    bool wait_for_start_key;
    //! The next key to send on completion
    uint32_t next_key;
    //! The key that indicates the start of simulation
    uint32_t start_key;
} workflow_t;

//! An input that receives spikes.
typedef struct {
    //! Information about the key and mask for the spike input
    key_info_t key_info;
    //! The maximum number of spikes to allow in the input
    uint32_t max_spikes;
    //! The number of components that the spike input targets
    uint32_t n_targets;
    //! The index of the components that the spike input targets
    uint32_t target_components[];
} spike_input_config_t;

//! An input that receives data from SDRAM using DMA
typedef struct {
    //! The address in SDRAM to read from
    uint32_t address;
    //! The size of the data to read, in bytes
    uint32_t size_in_bytes;
    //! The number of components that the SDRAM input targets
    uint32_t n_targets;
    //! The indices of the component that the SDRAM input targets
    uint32_t target_components[];
} sdram_input_config_t;

//! An output that sends data to SDRAM using DMA
typedef struct {
    //! The address in SDRAM to write to
    uint32_t address;
    //! The size of the data to write, in bytes
    uint32_t size_in_bytes;
    //! The source component of the output data to copy
    uint32_t component_index;
} sdram_output_config_t;

//! The configuration of a component in a workflow.
typedef struct {
    // The component to be executed in this step
    uint32_t component_id;
    // The size of the input data object for this component, in 32-bit words
    uint32_t input_size;
    // The size of the output data object for this component, in 32-bit words
    uint32_t output_size;
    // The size of the parameters for this component, in 32-bit words
    uint32_t param_size;
    // The number of inputs to this component
    uint32_t n_inputs;
    // The number of outputs from this component
    uint32_t n_outputs;
    // The indices of the next components that this component feeds into
    uint32_t outputs[];
    // The parameter data for this component follows; this has undefined length
    // void params[];
} workflow_component_config_t;

//! The configuration of a workflow, which consists of multiple components.
typedef struct {
    // The number of spike inputs to this component
    uint32_t n_spike_inputs;
    // The number of SDRAM inputs to this component
    uint32_t n_sdram_inputs;
    // The number of SDRAM outputs from this component
    uint32_t n_sdram_outputs;
    //! The number of components in the workflow
    uint32_t n_components;
    //! Whether to send the next key on completion
    bool send_next_key;
    //! Whether to wait for the start key to start
    bool wait_for_start_key;
    //! The next key to send on completion
    uint32_t next_key;
    //! The key that indicates the start of simulation
    uint32_t start_key;
    // A pointer to data to make finding the rest of the data easier
    uint32_t data[];
    // The spike inputs for this component follows
    // spike_input_config_t spike_inputs[n_spike_inputs];
    // The SDRAM inputs for this component follows
    // sdram_input_config_t sdram_inputs[n_sdram_inputs];
    // The SDRAM outputs for this component follows
    // sdram_output_config_t sdram_outputs[n_sdram_outputs];
    // The components in the workflow follows; this has undefined length
    // workflow_component_config_t components[];
} workflow_config_t;

//! \brief A union that can be used to identify a DMA operation
typedef union {
    //! The ID of the DMA operation
    uint32_t id;
    struct {
        //! The index of the element this is a transfer for
        uint32_t index: 31;
        //! Whether this is an input (1) or output (0)
        uint32_t is_input: 1;
    };
} dma_id_t;

//! \brief Configures a set of workflow components based on the provided
//!        configuration.
//! \param[in] config: Pointer to workflow configuration
//! \param[out] workflow: Pointer to a configured workflow pointer
//! \return True if the configuration was successful, false otherwise
bool configure_workflow(workflow_config_t *config, workflow_t **workflow);

//! \brief Processes a packet in the workflow.
//! \param[in] time: The current simulation time
//! \param[in] key: The key of the packet to process
//! \param[in] workflow: Pointer to a configured workflow
//! \return True if the packet was processed successfully, false otherwise
bool process_packet(uint32_t time, uint32_t key, workflow_t *workflow);

//! \brief Sets up the next read operation for the workflow.
//! \param[in] workflow: Pointer to a configured workflow
void setup_read_dma(workflow_t *workflow);

//! \brief Processes the result of a DMA completion.
//! \param[in] id: The ID of the DMA operation that completed
//! \param[in] workflow: Pointer to a configured workflow
void process_dma_complete(dma_id_t id, workflow_t *workflow);

//! \brief Runs a set of workflow components.  Note that the order is expected
//!        to have been determined before this call, and that all inputs and
//!        outputs are already allocated and mapped appropriately.
//! \param[in] workflow: Pointer to a configured workflow
void run_workflow(workflow_t *workflow);

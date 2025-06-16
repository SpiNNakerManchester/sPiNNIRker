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
    //! Pointers to data to copy from the output to the input
    void **data_to_copy;
    //! Output data pointer
    void *output;
} workflow_component_t;

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
    uint32_t next_components[];
    // The parameter data for this component follows
    // void params[];
} workflow_component_config_t;

//! The configuration of a workflow, which consists of multiple components.
typedef struct {
    //! Whether the output of the workflow is spikes
    bool output_spikes;
    //! Whether the output of the workflow is recorded or sent (only spikes)
    bool output_recorded;
    //! The base key to send spikes with if output_spikes is true and
    //! output_recorded is false
    uint32_t base_key;
    //! The number of components in the workflow
    uint32_t n_components;
    //! The components in the workflow
    workflow_component_config_t components[];
} workflow_config_t;

//! \brief Configures a set of workflow components based on the provided
//!        configuration.
//! \param[in] n_components: Number of components in the list
//! \param[in] config: Pointer to start of the configuration data
//! \param[out] output_components: Pointer to an array of configured components
//! \return True if the configuration was successful, false otherwise
bool configure_workflow(uint32_t n_components,
        workflow_component_config_t *config,
        workflow_component_t **output_components);

//! \brief Runs a set of workflow components.  Note that the order is expected
//!        to have been determined before this call, and that all inputs and
//!        outputs are already allocated and mapped appropriately.
//! \param n_components: Number of components in the workflow
//! \param components: Array of workflow components to execute in order
void run_workflow(uint32_t n_components, workflow_component_t *components);

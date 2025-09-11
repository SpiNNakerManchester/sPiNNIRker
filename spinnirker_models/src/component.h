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
#ifndef __component_h__
#define __component_h__

//! \file component.h
//! \brief Workflow component definitions
#include <stdint.h>

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

typedef enum {
    //! Data that is a matrix of input values that is always the same size.
    //! Note that the type of values is determined by the component.
    DATA_TYPE_MATRIX = 0,
    //! DAta that is a list of spikes
    DATA_TYPE_SPIKES = 1,
} data_type_t;

typedef struct {
    //! The type of the data
    data_type_t type;
    //! The data itself
    void *data;
} data_t;

//! \brief Defines a function type for components in a workflow.
//! Note no input is the same as the output.
//! \param[in] data: Pointer to the workflow data structure that holds
//!                  the state and configuration for the workflow
//! @param[in] n_inputs: Number of input data pointers for the component
//! \param[in] input: Input data for the component
//! \param[out] output: Output data for the component
typedef void (*component_func)(void *data, uint32_t n_inputs, data_t *input,
        data_t output);

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

#endif // __component_h__

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

extern void spin1_wfi(void);

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

//! \brief A union that can be used to identify a DMA operation
typedef union {
    //! The ID of the DMA operation
    uint32_t id;
    struct {
        //! The index of the element / component this is a transfer for
        uint32_t index: 30;
        //! Whether this is for a component (1) or an input/output (0)
        uint32_t is_component: 1;
        //! Whether this is an input (1) or output (0)
        uint32_t is_input: 1;
    };
} dma_id_t;

//! Structure for constants for precise constant integer division (see div_by_const)
typedef struct {
    uint32_t m: 16;
    uint32_t sh1: 8;
    uint32_t sh2: 8;
} div_const;

//! \brief Divide by a constant - based on https://doi.org/10.1145/178243.178249
static inline uint32_t div_by_const(uint32_t i, div_const d) {
    uint32_t t1 = (i * d.m) >> 16;
    uint32_t isubt1 = (i - t1) >> d.sh1;
    return (t1 + isubt1) >> d.sh2;
}

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
//! \param[in] index: The index of the component in the workflow
//! \param[in] params: Pointer to parameters for the component
//! \return Pointer to the initialized component data structure
typedef void* (*component_init)(uint32_t index, void *params);

//! \brief Defines a function type for handling DMA completion for a component
//! \param[in] tag: The DMA ID tag that has completed
//! \param[in] data: Pointer to the component data structure
typedef void (*component_dma_complete)(dma_id_t tag, void *data);

//! A component that could be used in a workflow.
typedef struct {
    //! Function to call for this component
    component_func func;
    //! Function to call to initialize this component
    component_init init;
    //! Function to call when a DMA is complete for this component (can be NULL)
    component_dma_complete dma_complete;
} component_t;

#endif // __component_h__

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
//! \file leaky_integrator_common.h
//! \brief leaky_integrator NIR component common code
#include <stdfix-full-iso.h>

typedef struct {
    int32_t decay;       //!< The decay factor - signed long fract
    int32_t resistance;  //!< The resistance factor - signed long accum
} leaky_integrator_item_t;

static inline int32_t decay(int32_t s1615_value, uint32_t decay) {
    return __stdfix_sat_k((__I64(s1615_value) * __U64(decay)) >> 32);
}

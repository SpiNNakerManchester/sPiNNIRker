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
//! \file decay.h
//! \brief decay common code
#include <stdfix-full-iso.h>
#include "decay.h"

static inline int32_t decay_multi(int32_t s1615_value, uint32_t decay_val,
        uint32_t n_steps) {
    uint32_t result = 1;
    uint32_t next_decay = decay_val;
    uint32_t step = n_steps;
    while (step > 0) {
        if (step & 1) {
            result = __stdfix_smul_ulr(result, next_decay);
        }
        step >>= 1;
        if (step > 0) {
            next_decay = __stdfix_smul_ulr(next_decay, next_decay);
        }
    }
    return decay(s1615_value, result);
}

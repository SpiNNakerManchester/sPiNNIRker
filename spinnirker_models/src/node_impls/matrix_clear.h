/*
 * Copyright (c) 2025 The University of Manchester
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

#include "../component.h"
static void matrix_clear_outputs(data_t output, uint32_t n_words) {
    // Convert to right type for output (Accum but only ever added to)
    uint32_t *out_data = output.data;

    // Clear the output
    for (uint32_t i = 0; i < n_words; i++) {
        out_data[i] = 0;
    }
}

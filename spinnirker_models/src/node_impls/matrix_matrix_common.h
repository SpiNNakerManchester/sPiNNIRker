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
//! \file matrix_matrix_common.h
//! \brief Common matrix functions when the input is matrices

#include "matrix_common.h"

typedef struct {
    //! The information about the matrix being processed
    matrix_data_t *data;

    //! The current index of the row being processed
    uint32_t current_row;

    //! The current index of the column being processed
    uint32_t current_col;

    //! The current row of the matrix that has been transferred
    void *current_data;
} matrix_loop_t;

extern matrix_loop_t matrix_loop_start(matrix_data_t *matrix);

extern uint32_t matrix_loop_is_next(matrix_loop_t *loop,
    uint32_t *row, uint32_t *col);

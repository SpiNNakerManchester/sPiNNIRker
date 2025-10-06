# Copyright (c) 2025 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

APP = run_workflow
SOURCES = workflow.c main.c \
	node_impls/matrix_common_init.c \
	node_impls/matrix_matrix_common.c node_impls/matrix_spikes_common.c \
	node_impls/input.c \
	node_impls/output_matrix.c node_impls/output_spikes.c \
	node_impls/linear_matrix.c node_impls/linear_spikes.c \
	node_impls/affine_matrix.c node_impls/affine_spikes.c \
	node_impls/scale_matrix.c node_impls/scale_spikes.c \
	node_impls/spike_matrix.c node_impls/spike_spikes.c \
	node_impls/integrator_matrix.c node_impls/integrator_spikes.c \
	node_impls/leaky_integrator_matrix.c node_impls/leaky_integrator_spikes.c \
	node_impls/integrate_and_fire_matrix.c node_impls/integrate_and_fire_spikes.c \
	node_impls/leaky_integrate_and_fire_matrix.c node_impls/leaky_integrate_and_fire_spikes.c \
	node_impls/current_based_leaky_integrator_matrix.c node_impls/current_based_leaky_integrator_spikes.c

include ../support.mk

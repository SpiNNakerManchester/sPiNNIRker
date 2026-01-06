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

# If FEC_INSTALL_DIR is not defined, this is an error!
FEC_INSTALL_DIR := $(strip $(if $(FEC_INSTALL_DIR), $(FEC_INSTALL_DIR), $(if $(SPINN_DIRS), $(SPINN_DIRS)/fec_install, $(error FEC_INSTALL_DIR or SPINN_DIRS is not set.  Please define FEC_INSTALL_DIR or SPINN_DIRS))))

# Define the directories
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MODELS_DIR := $(abspath $(dir $(MAKEFILE_PATH))/../)/
SPINNIRKER_DIR := $(abspath $(dir $(MAKEFILE_PATH))/../../)/

SRC_DIR := $(MODELS_DIR)src/
MODIFIED_DIR := $(MODELS_DIR)modified_src/
SOURCE_DIRS += $(SRC_DIR):$(MODIFIED_DIR)
BUILD_DIR := $(MODELS_DIR)builds/$(APP)/
APP_OUTPUT_DIR := $(SPINNIRKER_DIR)spinnirker/model_binaries/

include $(FEC_INSTALL_DIR)/make/fec.mk

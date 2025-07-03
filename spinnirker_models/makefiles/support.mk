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

# If SPINN_DIRS is not defined, this is an error!
ifndef SPINN_DIRS
    $(error SPINN_DIRS is not set.  Please define SPINN_DIRS (possibly by running "source setup" in the spinnaker package folder))
endif
MAKEFILE_PATH := $(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# APP name for a and dict files
ifndef APP
    $(error APP is not set.  Please define APP)
endif

# Define the directories
SRC_DIR := $(abspath $(MAKEFILE_PATH)/../src/)
SOURCE_DIRS += $(SRC_DIR)
MODIFIED_DIR := $(abspath $(MAKEFILE_PATH)/../modified_src/)
BUILD_DIR := $(abspath $(MAKEFILE_PATH)/../builds/$(APP)/)
APP_OUTPUT_DIR := $(abspath $(MAKEFILE_PATH)/../../spinnirker/model_binaries/)

include $(SPINN_DIRS)/make/local.mk


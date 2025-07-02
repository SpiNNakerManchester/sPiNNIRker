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

from nir import NIRGraph

from pacman.model.graphs.machine import MachineVertex
from pacman.model.graphs.common import Slice
from pacman.model.placements import Placement
from pacman.model.resources import AbstractSDRAM

from spinnman.model.enums import ExecutableType

from spinn_front_end_common.abstract_models import (
    AbstractHasAssociatedBinary, AbstractGeneratesDataSpecification)
from spinn_front_end_common.interface.ds import DataSpecificationGenerator

from .workflow_application_vertex import WorkflowApplicationVertex
from .subgraph import SubGraph


class WorkflowMachineVertex(
        MachineVertex, AbstractHasAssociatedBinary,
        AbstractGeneratesDataSpecification):
    """ A machine vertex of a workflow element.
    """

    __slots__ = ()

    def __init__(
            self, label: str, nir_model: NIRGraph, subgraph: SubGraph,
            app_vertex: WorkflowApplicationVertex, vertex_slice: Slice):
        MachineVertex.__init__(
            self, label=label, app_vertex=app_vertex,
            vertex_slice=vertex_slice)

    @property
    def sdram_required(self) -> AbstractSDRAM:
        # TODO:
        return MachineVertex.sdram_required(self)

    def get_binary_file_name(self) -> str:
        # TODO: Determine if this needs to change depending on workflow
        # components
        return "run_workflow.aplx"

    def get_binary_start_type(self) -> ExecutableType:
        return ExecutableType.USES_SIMULATION_INTERFACE

    def generate_data_specification(
            self, spec: DataSpecificationGenerator,
            placement: Placement) -> None:
        # TODO:
        AbstractGeneratesDataSpecification.generate_data_specification(
            self, spec, placement)

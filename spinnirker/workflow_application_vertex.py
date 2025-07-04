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
from spinn_utilities.overrides import overrides

from pacman.model.graphs.application import ApplicationVertex


class WorkflowApplicationVertex(ApplicationVertex):
    """ An application vertex of a workflow element.
    """

    __slots__ = ()

    def __init__(self, label: str):
        """

        :param label: The label of the vertex
        """
        ApplicationVertex.__init__(self, label=label)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self) -> int:
        return 0

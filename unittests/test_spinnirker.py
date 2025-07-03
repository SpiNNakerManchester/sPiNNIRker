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
from nir import NIRGraph, LI, Threshold
from spinnirker import get_node_connection_maps, split_graph, SubGraph
from numpy import array


def test_split_graph() -> None:
    # Test graph gets split correctly at the B2 and B6 Threshold nodes.
    # R1 → B2 → R3 → R4
    # ↓         ↑
    # R5 → B6 → R7
    nir_graph = NIRGraph(
        nodes={
            "R1": LI(array([0]), array([0]), array([0])),
            "B2": Threshold(array([0])),
            "R3": LI(array([0]), array([0]), array([0])),
            "R4": LI(array([0]), array([0]), array([0])),
            "R5": LI(array([0]), array([0]), array([0])),
            "B6": Threshold(array([0])),
            "R7": LI(array([0]), array([0]), array([0]))
        },
        edges=[
            ("R1", "B2"),
            ("B2", "R3"),
            ("R3", "R4"),
            ("R1", "R5"),
            ("R5", "B6"),
            ("B6", "R7"),
            ("R7", "R3")
        ]
    )

    _, outgoing_map = get_node_connection_maps(nir_graph)
    subgraphs = split_graph(nir_graph, outgoing_map)

    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        assert isinstance(subgraph, SubGraph)
        assert subgraph.nodes in (
            {"R1", "B2", "R5", "B6"}, {"R3", "R4", "R7"})

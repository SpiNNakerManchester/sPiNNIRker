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
from collections import defaultdict

from typing import List, Tuple, Dict, Type

from spinn_utilities.ordered_set import OrderedSet

from nir import NIRGraph, NIRNode, LIF, IF, CubaLIF, Threshold

from .subgraph import SubGraph


def get_node_connection_maps(nir_model: NIRGraph) -> Tuple[
        Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get maps for incoming and outgoing connections for all nodes in the
    NIR graph.

    :param nir_model: The NIR graph model containing the nodes and edges

    :return:
        A map of node names to lists of incoming nodes and
        a map of node names to lists of outgoing nodes.
    """
    incoming_map = defaultdict(list)
    outgoing_map = defaultdict(list)
    for source, target in nir_model.edges:
        incoming_map[target].append(source)
        outgoing_map[source].append(target)
    return incoming_map, outgoing_map


def split_graph(
        nir_model: NIRGraph,
        outgoing_map: Dict[str, List[str]],
        split_classes: Tuple[Type[NIRNode], ...] = (
            Threshold, CubaLIF, LIF, IF)) -> OrderedSet[SubGraph]:
    """
    Split the NIR graph into sub-graphs that communicate with each other
    only via a Threshold node i.e. a node that sends only 1 or 0 values (where
    sending a 0 is assumed to mean sending nothing here).  Note that this might
    split the graph into a single sub-graph.

    @param nir_model:
        The NIR graph model containing the nodes and edges
    @param outgoing_map:
        A map of nodes to lists of nodes that they target
    @return:
        A list of the nodes in each sub-graph
    """

    # Keep a list of sub-graphs, and which node is in which graph
    subgraph_nodes: OrderedSet[SubGraph] = OrderedSet()
    node_subgraph_map: Dict[str, SubGraph] = dict()

    # Start with list of unvisited nodes, to make sure we visit them all!
    unvisited = OrderedSet(outgoing_map.keys())
    while unvisited:
        first_subgraph: SubGraph = SubGraph()

        # Do a traversal starting at one node, and add nodes found to the
        # current sub-graph.
        # When we hit a Threshold node, we create new sub-graphs for each node
        # that the Threshold node targets.
        # If we hit a node already in a separate sub-graph, we merge the
        # sub-graphs together.
        first_node: str = unvisited.pop()
        nodes: List[Tuple[str, SubGraph]] = [(first_node, first_subgraph)]
        while nodes:
            node, subgraph = nodes.pop()
            unvisited.discard(node)

            # This is a new node so explore!
            if node not in node_subgraph_map:
                # Add the node to the current sub-graph
                subgraph.nodes.add(node)
                subgraph_nodes.add(subgraph)
                # Mark the node as being in the current sub-graph
                node_subgraph_map[node] = subgraph

                # Push all target nodes to the node list
                for target in outgoing_map[node]:
                    if isinstance(nir_model.nodes[node], split_classes):
                        # If the current node is a Threshold node, create a new
                        # sub-graph for the target
                        nodes.append((target, SubGraph()))
                    else:
                        # Otherwise just add it with the current sub-graph
                        nodes.append((target, subgraph))

            # Otherwise check if the node is actually in a different sub-graph
            # from the expected one
            elif node_subgraph_map[node] != subgraph:
                # If it is different, merge the two sub-graphs
                old_subgraph = node_subgraph_map[node]
                for n in old_subgraph.nodes:
                    node_subgraph_map[n] = subgraph
                    subgraph.nodes.add(n)
                # Remove the old sub-graph from the list of sub-graphs
                subgraph_nodes.discard(old_subgraph)
                subgraph_nodes.add(subgraph)

    return subgraph_nodes

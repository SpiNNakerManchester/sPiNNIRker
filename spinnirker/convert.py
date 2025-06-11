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

import logging
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Dict, Set

from nir import (
    NIRGraph, NIRNode, Affine, Conv1d, Conv2d, Linear, LIF, IF, CubaLIF, Input,
    Threshold)
from numpy.typing import NDArray
from numpy import abs, max, newaxis, zeros, float64

from spynnaker.pyNN import Population, IF_curr_exp, IF_curr_delta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NEURON_NODES = (LIF, IF, CubaLIF)
SPIKE_SOURCE_NODES = (LIF, IF, CubaLIF, Input)
WEIGHT_NODES = (Affine, Conv1d, Conv2d, Linear)


DT: float = 1.0
OUTPUT_RECORD: List[str] = ["spikes"]
DELAY: int = 1

def get_accumulated_bias(
        incoming_nodes: List[NIRNode],
        input_shape: Tuple[int, ...]) -> NDArray[float64]:
    """
    Get the accumulated bias for all inputs to a node.

    This function is applied to neuron nodes such as `LIF` or `CubaLIF` and
    looks for incoming nodes with bias (currently: `Affine`, `Conv1d` and
    `Conv2d`). From these incoming nodes the biases are accumulated and
    returned.

    For Affine nodes, the bias typically has a 1D shape that matches the shape
    of the subsequent neuron node.

    For Conv nodes, the bias per output channel is broadcasted to match the
    target node's input shape.

    @param incoming_nodes: List of incoming nodes to the target node.
    @param input_shape: Shape of the target node's input.

    @return Accumulated bias with the same shape as the target node.
    """
    bias = zeros(input_shape)

    # TODO: Understand more about the input_shape and the bias shape.
    for in_node in incoming_nodes:
        if isinstance(in_node, Affine):
            bias += in_node.bias
        elif isinstance(in_node, Conv1d):
            bias += in_node.bias[:, newaxis]
        elif isinstance(in_node, Conv2d):
            bias += in_node.bias[:, newaxis, newaxis]

    return bias.flatten()


def get_max_abs_weight(incoming_nodes: List[NIRNode]) -> float:
    """
    Get the maximum absolute weight from all incoming synapses.

    This function is applied to neuron nodes such as `LIF` or `CubaLIF` and
    looks for incoming nodes with weights (currently: `Affine`, `Linear`,
    `Conv1d` and `nir.Conv2d`). From theses incoming nodes the maximum weight
    is determined and returned.

    @param incoming_nodes: List of incoming nodes to the target node.

    @return maximum absolute weight
    """
    max_weight = 0.0
    for in_node in incoming_nodes:
        if isinstance(in_node, WEIGHT_NODES):
            max_weight_node = abs(in_node.weight).max()
            max_weight = max((max_weight, max_weight_node))
    return max_weight


def convert_LIF(name: str, node: LIF, bias: NDArray[float64]) -> Population:
    """
    Convert the LIF parameters to PyNN.

    @param name The name of the node
    @param node The node to convert
    @param bias bias array with same shape as the input of the node

    @return: PyNN Population with the converted parameters.
    """

    model = IF_curr_delta(
        v_rest=node.v_leak,
        v_reset=node.v_leak,
        v_thresh=node.v_threshold,
        tau_m=node.tau,
        cm=(node.tau / node.r),
        tau_refrac=0.0,
        i_offset=bias)
    pop = Population(node.r.shape, model, label=name)
    return pop

def convert_CubaLIF(
        name: str, node: CubaLIF, bias: NDArray[float64]) -> Population:
    """
    Convert the LIF parameters to PyNN.

    @param name The name of the node
    @param node The node to convert
    @param bias bias array with same shape as the input of the node

    @return: PyNN Population with the converted parameters.
    """

    model = IF_curr_exp(
        v_rest=node.v_leak,
        v_reset=node.v_leak,
        v_thresh=node.v_threshold,
        tau_m=node.tau_mem,
        cm=(node.tau_mem / node.r),
        tau_refrac=0.0,
        tau_syn_E=node.tau_syn,
        tau_syn_I=node.tau_syn)
    pop = Population(node.r.shape, model, label=name)
    return pop


def convert_IF(node: nir.NIRNode, bias: np.ndarray, config: ConversionConfig, w_scale: float):  # noqa: N802
    """convert the IF parameters to SpiNNaker2.

    Args:
        node: NIR node
        bias: bias array with same shape as node
        config: NIR-to-SpiNNaker2-conversion configuration
        w_scale: factor by which weights are scaled. Will be applied to
            parameters `threshold` and`i_offset`.

    Returns:
        tuple: (neuron_params, v_scale)

        `neuron_params` contains the parameters for the spinnaker2
        `lif_no_delay` population, while `v_scale` is the factor by which the
        threshold was scaled during translation from NIR to SpiNNaker2.
    """
    assert isinstance(node, nir.IF)
    bias_factor = node.r.flatten()
    v_scale = 1.0 / bias_factor * w_scale

    neuron_params = {
        "threshold": node.v_threshold.flatten() * v_scale,
        "alpha_decay": 1.0,
        "i_offset": bias * w_scale,
        "reset": get_reset_method(config.reset),
    }
    return neuron_params, v_scale


def create_populations(nir_model, config: ConversionConfig):
    populations = []
    input_populations = []
    output_populations = []
    for name, node in nir_model.nodes.items():
        print(f"Node '{name}'")
        print(f"Got {type(node)}")
        if isinstance(node, (nir.LIF, nir.IF, nir.CubaLIF)):
            bias = get_accumulated_bias(name, nir_model)
            w_scale = 1.0
            if config.scale_weights:
                if config.weight_scale_percentile == 100:
                    max_abs_weight = get_max_abs_weight(name, nir_model)
                else:
                    max_abs_weight = get_percentile_abs_weight(config.weight_scale_percentile, name, nir_model)
                w_scale = 127.0 / max_abs_weight  # TODO: 127 should not be hard-coded
            if any([isinstance(pre_node, nir.Conv2d) for _, pre_node in get_incoming_nodes(name, nir_model)]):
                is_conv2d = True
            else:
                is_conv2d = False

            if isinstance(node, nir.LIF):
                neuron_model_name = "lif_conv2d" if is_conv2d else "lif_no_delay"
                neuron_params, v_scale = convert_LIF(node, bias, config, w_scale)
            elif isinstance(node, nir.IF):
                neuron_model_name = "lif_conv2d" if is_conv2d else "lif_no_delay"
                neuron_params, v_scale = convert_IF(node, bias, config, w_scale)
            else:  # CubaLIF
                print("Got CubaLIF!")
                assert is_conv2d == False, "Conv2d with CubaLIF is currently not supported!"
                neuron_model_name = "lif_curr_exp_no_delay"
                neuron_params, v_scale = convert_CubaLIF(node, bias, config, w_scale)

            if any([isinstance(out_node, nir.Output) for _, out_node in get_outgoing_nodes(name, nir_model)]):
                record = config.output_record
                print(record)
                has_output = True
            else:
                record = []
                has_output = False

            print("record: ", record)
            input_shape = node.input_type["input"]
            print(input_shape, "->", node.output_type["output"])
            pop = snn.Population(
                size=np.prod(input_shape),  # TODO what happens for 2D shape?
                neuron_model=neuron_model_name,
                params=neuron_params,
                name=name,
                record=record,
            )
            pop.nir_v_scale = v_scale  # save scaling factors to rescale recorded voltages
            pop.nir_w_scale = w_scale  # weight scale needed later for creation of projections
            populations.append(pop)
            if has_output:
                output_populations.append(pop)

        elif isinstance(node, nir.Input):
            # infer shape
            input_shape = node.input_type["input"]
            assert input_shape.size == 1 or input_shape.size == 3, "only 1d or 3d input allowed"
            pop = snn.Population(size=np.prod(input_shape), neuron_model="spike_list", params={}, name=name)
            populations.append(pop)
            input_populations.append(pop)
        elif isinstance(node, nir.Output):
            pass
        elif isinstance(node, WEIGHT_NODES):
            # TODO: check for any other combinations?
            for _, target_node in get_connected_nodes(name, nir_model):
                if isinstance(target_node, WEIGHT_NODES):
                    raise HardwareConstraintError(
                        f"Two successive layers with weights are not "
                        f"supported! Got: {type(node)} and {type(target_node)}"
                    )
        elif isinstance(node, nir.SumPool2d):
            pass
        elif isinstance(node, nir.Flatten):
            pass
        else:
            print("Got", type(node))
            raise NotImplementedError(type(node))
        print("input: ", [(e, type(n)) for e, n in get_incoming_nodes(name, nir_model)])
        print("output: ", [(e, type(n)) for e, n in get_outgoing_nodes(name, nir_model)])
        print("")
    return populations, input_populations, output_populations


def create_projection(origin_name, target_name, affine_node, populations, delay):
    assert isinstance(affine_node, (nir.Affine, nir.Linear))
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    conns = ann2snn_helpers.connection_list_from_dense_weights(affine_node.weight.T * post.nir_w_scale, delay)
    proj = snn.Projection(pre=pre, post=post, connections=conns)
    return proj


def get_conv2d_params(conv2d_node, post_node):
    input_shape = conv2d_node.input_type["input"]
    conv_weights = conv2d_node.weight.swapaxes(0, 1) * post_node.nir_w_scale
    conv_weights = conv_weights.astype(np.int8)
    conv_params = {
        "in_height": input_shape[1],
        "in_width": input_shape[2],
        "stride_x": nir.ir.utils._index_tuple(conv2d_node.stride, 0),
        "stride_y": nir.ir.utils._index_tuple(conv2d_node.stride, 1),
        "pool_x": 1,
        "pool_y": 1,
        "pad_top": nir.ir.utils._index_tuple(conv2d_node.padding, 0),
        "pad_bottom": nir.ir.utils._index_tuple(conv2d_node.padding, 0),
        "pad_left": nir.ir.utils._index_tuple(conv2d_node.padding, 1),
        "pad_right": nir.ir.utils._index_tuple(conv2d_node.padding, 1),
    }
    return conv_params, conv_weights


def create_conv2d_projection(origin_name, target_name, conv2d_node, populations, delay):
    assert delay == 0, f"Conv2DProjection do not support a delay differnt from 1"
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    conv_params, conv_weights = get_conv2d_params(conv2d_node, post)
    print(conv_params, "weights: ", conv_weights.shape)

    proj = snn.Conv2DProjection(pre, post, conv_weights, conv_params)

    return proj


def create_sumpool2d_projection(origin_name, target_name, sumpool2d_node, populations, delay):
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    # C_out, C_in, H, W
    sumpool2d_conns, sumpool2d_output_shape = ann2snn_helpers.connection_list_for_sumpool2d(
        input_shape=sumpool2d_node.input_type["input"],
        stride=sumpool2d_node.stride,
        kernel_size=sumpool2d_node.kernel_size,
        padding=sumpool2d_node.padding,
        delay=delay,
        data_order="torch",
    )

    proj = snn.Projection(pre=pre, post=post, connections=sumpool2d_conns)
    return proj


def create_sumpool2d_conv2d_projection(origin_name, target_name, sumpool2d_node, conv2d_node, populations, delay):
    assert delay == 0, f"Conv2DProjection do not support a delay differnt from 1"
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)

    conv_params, conv_weights = get_conv2d_params(conv2d_node, post)
    print(conv_params, "weights: ", conv_weights.shape)
    print(sumpool2d_node)
    assert all(sumpool2d_node.kernel_size == sumpool2d_node.stride)
    assert nir.ir.utils._index_tuple(sumpool2d_node.padding, 0) == 0
    assert nir.ir.utils._index_tuple(sumpool2d_node.padding, 1) == 0
    conv_params["pool_x"] = nir.ir.utils._index_tuple(sumpool2d_node.kernel_size, 0)
    conv_params["pool_y"] = nir.ir.utils._index_tuple(sumpool2d_node.kernel_size, 1)
    # fix input sizes (we need to take the ones from the sumpool2d_node)
    input_shape = sumpool2d_node.input_type["input"]
    conv_params["in_height"] = input_shape[1]
    conv_params["in_width"] = input_shape[2]

    proj = snn.Conv2DProjection(pre, post, conv_weights, conv_params)

    return proj


def create_sumpool2d_affine_projection(origin_name, target_name, sumpool2d_node, affine_node, populations, delay):
    # this might be a bit hacky, but should work...
    pre = fetch_population_by_name(origin_name, populations)
    post = fetch_population_by_name(target_name, populations)
    # C_out, C_in, H, W
    sumpool2d_conns, sumpool2d_output_shape = ann2snn_helpers.connection_list_for_sumpool2d(
        input_shape=sumpool2d_node.input_type["input"],
        stride=sumpool2d_node.stride,
        kernel_size=sumpool2d_node.kernel_size,
        padding=sumpool2d_node.padding,
        delay=delay,
        data_order="torch",
    )
    affine_conns = ann2snn_helpers.connection_list_from_dense_weights(
        affine_node.weight.T * post.nir_w_scale, delay=delay
    )
    conns = ann2snn_helpers.join_conn_lists(sumpool2d_conns, affine_conns)

    proj = snn.Projection(pre=pre, post=post, connections=conns)
    return proj


def create_projections(nir_model, populations, delay=1):
    logger.debug("creating projections(): start")
    projections = []
    edges = nir_model.edges

    for edge in edges:
        logger.debug(f"checking {edge[0]}->{edge[1]}")
        origin = nir_model.nodes[edge[0]]
        if isinstance(origin, SPIKE_SOURCE_NODES):
            target_name = edge[1]
            target = nir_model.nodes[target_name]
            if isinstance(target, (nir.Affine, nir.Linear)):
                logger.debug(f"  found {type(target)}, next search for neuron node")
                final_targets = get_outgoing_nodes(target_name, nir_model)
                for final_target_name, final_target in final_targets:
                    logger.debug(f"    found final target {final_target_name}")
                    assert isinstance(final_target, NEURON_NODES)
                    logger.debug(f"    create projection between {edge[0]} and {final_target_name}")
                    proj = create_projection(edge[0], final_target_name, target, populations, delay)
                    projections.append(proj)
            elif isinstance(target, nir.SumPool2d):
                logger.debug("  found SumPool2d, next search for path to neurons")
                sumpooltargets = get_outgoing_nodes(target_name, nir_model)
                for sumpooltarget_name, sumpooltarget in sumpooltargets:
                    if isinstance(sumpooltarget, nir.Conv2d):
                        logger.debug(f"    found Conv2d: {sumpooltarget_name}")
                        final_targets = get_outgoing_nodes(sumpooltarget_name, nir_model)
                        for final_target_name, final_target in final_targets:
                            logger.debug(f"    found final target {final_target_name}")
                            assert isinstance(final_target, NEURON_NODES)
                            logger.debug(
                                f"    create sumpool2d_convd_projection between {edge[0]} and {final_target_name}"
                            )
                            proj = create_sumpool2d_conv2d_projection(
                                edge[0],
                                final_target_name,
                                target,
                                sumpooltarget,
                                populations,
                                delay,
                            )
                            projections.append(proj)
                    elif isinstance(sumpooltarget, NEURON_NODES):
                        logger.debug(f"    found Neuron: {sumpooltarget_name}")
                        final_targets = get_outgoing_nodes(sumpooltarget_name, nir_model)
                        nodetype = "IF" if isinstance(sumpooltarget, nir.IF) else "LIF"
                        logger.debug(f"    create sumpool2d_projection between {edge[0]} and {sumpooltarget_name}")
                        proj = create_sumpool2d_projection(edge[0], sumpooltarget_name, target, populations, delay)
                        projections.append(proj)
                    elif isinstance(sumpooltarget, nir.Flatten):
                        logger.debug(f"    found Flatten: {sumpooltarget_name}")
                        flatten_targets = get_outgoing_nodes(sumpooltarget_name, nir_model)
                        for flatten_target_name, flatten_target in flatten_targets:
                            if isinstance(flatten_target, nir.Affine):
                                final_targets = get_outgoing_nodes(flatten_target_name, nir_model)
                                for final_target_name, final_target in final_targets:
                                    logger.debug(f"    found final target {final_target_name}")
                                    assert isinstance(final_target, NEURON_NODES)
                                    logger.debug(
                                        f"    create sumpool2d_affine_projection between {edge[0]} and "
                                        f"{final_target_name}"
                                    )
                                    proj = create_sumpool2d_affine_projection(
                                        edge[0],
                                        final_target_name,
                                        target,
                                        flatten_target,
                                        populations,
                                        delay,
                                    )
                                    projections.append(proj)
                            else:
                                raise (
                                    NotImplementedError(
                                        "Currently after SumPool2d->Flatten, Affine has to follow! Other combinations "
                                        "not supported yet!"
                                    )
                                )
                    else:
                        raise (
                            NotImplementedError(
                                "Currently SumPool2d can only be connected to Conv2d or Flatten, others are not "
                                "supported yet!"
                            )
                        )
            elif isinstance(target, nir.Conv2d):
                logger.debug("  found Conv2d, next search for path to neurons")
                final_targets = get_outgoing_nodes(target_name, nir_model)
                for final_target_name, final_target in final_targets:
                    logger.debug(f"    found final target {final_target_name}")
                    assert isinstance(final_target, NEURON_NODES)
                    logger.debug(f"    create conv2d_projection between {edge[0]} and {final_target_name}")
                    proj = create_conv2d_projection(edge[0], final_target_name, target, populations, delay)
                    projections.append(proj)
            elif isinstance(target, NEURON_NODES):
                raise (
                    NotImplementedError(
                        f"Direct connections from spike source nodes ({SPIKE_SOURCE_NODES}) to neuron nodes "
                        f"({NEURON_NODES}) not supported yet!"
                    )
                )
            else:
                logger.debug("  discard edge as target does not represent a real connection.")
        else:
            logger.debug("  discard edge as source is not spike source node")
    return projections


def from_nir(nir_model: nir.NIRGraph, config: ConversionConfig = None):
    """create SpiNNaker2 network from NIR graph.

    Args:
        nir_model: NIR graph
        config: NIR-to-SpiNNaker2-conversion configuration
    Returns:
        tuple of length 3: (net, input_pops, output_pops)

        Details:
            net(snn.Network): SpiNNaker2 Network
            input_pops(list[snn.Population]): list of input populations
            output_pops(list[snn.Population]): list of output populations
    """
    assert isinstance(nir_model, nir.NIRGraph)
    if config == None:
        config = ConversionConfig()
    logger.info("from_nir(): create spinnaker2.Network from NIR graph")
    populations, input_populations, output_populations = create_populations(nir_model, config)
    logger.info(f"Created {len(populations)} populations: {[(_.name) for _ in populations]}")
    projections = create_projections(nir_model, populations, delay=config.conn_delay)
    logger.info(f"Created {len(projections)} projections: {[(_.name) for _ in projections]}")

    net = snn.Network()
    net.add(*populations, *projections)
    net.validate()
    return net, input_populations, output_populations



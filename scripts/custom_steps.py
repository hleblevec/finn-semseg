import numpy as np
import warnings
import math
from onnx import TensorProto, helper

from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat  
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    SortGraph,
    RemoveUnusedTensors,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
)
from qonnx.transformation.base import Transformation
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import (
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MoveIdenticalOpPastJoinOp,
    )
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO


from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    VerificationStepType,
    ShellFlowType,
)
from finn.builder.build_dataflow_steps import verify_step

from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from qonnx.util.config import extract_model_config_to_json


class MakeUpsampleNHWC(Transformation):
    """
    Converts the inputs and outputs for all Upsample nodes
    from NCHW to NHWC.
    """
    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Upsample":
                consumer = model.find_consumer(n.output[0])
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        old_value = model.get_initializer(n.input[1])
                        new_value = np.array([old_value[idx] for idx in (0, 2, 3, 1)], dtype=np.dtype('float32'))
                        model.set_initializer(n.input[1], new_value)
                        start_name = producer.input[0]
                        mid_name = n.input[0]
                        end_name = n.output[0]
                        (b, hi, wi, c) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(end_name)
                        producer.input[0] = mid_name
                        producer.output[0] = end_name
                        n.input[0] = start_name
                        n.output[0] = mid_name
                        model.set_tensor_shape(mid_name, (b, ho, wo, c))
                        model.set_tensor_shape(end_name, (b, c, ho, wo))
                        graph.node.remove(producer)
                        graph.node.insert(node_ind, producer)
                elif consumer is not None and consumer.op_type == "Transpose":
                    perms = list(get_by_name(consumer.attribute, "perm").ints)
                    if perms == [0, 2, 3, 1]:
                        old_value = model.get_initializer(n.input[1])
                        new_value = np.array([old_value[idx] for idx in (0, 2, 3, 1)], dtype=np.dtype('float32'))
                        model.set_initializer(n.input[1], new_value)
                        start_name = n.input[0]
                        mid_name = consumer.input[0]
                        end_name = consumer.output[0]
                        (b, c, hi, wi) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(mid_name)
                        consumer.input[0] = start_name
                        consumer.output[0] = mid_name
                        n.input[0] = mid_name
                        n.output[0] = end_name
                        model.set_tensor_shape(mid_name, (b, hi, wi, c))
                        model.set_tensor_shape(end_name, (b, ho, wo, c))
                        graph.node.remove(consumer)
                        graph.node.insert(node_ind - 1, consumer)
        return (model, False)

class CustomMakeMaxPoolNHWC(Transformation):
    """Convert (MaxPool, NHWCTranspose) into (NHWCTranspose, MaxPoolNHWC)
    and (NCHWTranspose, MaxPool) into (MaxPoolNHWC, NCHWTranspose)."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        fork_node = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MaxPool":
                if model.is_fork_node(n):
                    fork_node = True
                    consumers = model.find_consumers(n.output[0])
                    producer = model.find_producer(n.input[0])
                    for consumer_node in consumers:
                        if consumer_node.op_type == "Transpose":
                            consumer = consumer_node
                            break
                else:
                    consumer = model.find_consumer(n.output[0])
                    producer = model.find_producer(n.input[0])
                if consumer is not None and consumer.op_type == "Transpose":
                    perms = list(get_by_name(consumer.attribute, "perm").ints)
                    if perms == [0, 2, 3, 1]:
                        n.op_type = "MaxPoolNHWC"
                        n.domain = "qonnx.custom_op.general"
                        start_name = n.input[0]
                        mid_name = consumer.input[0]
                        end_name = consumer.output[0]
                        (b, c, hi, wi) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(mid_name)
                        consumer.input[0] = start_name
                        consumer.output[0] = mid_name
                        n.input[0] = mid_name
                        n.output[0] = end_name
                        if fork_node:
                            for consumer_node in consumers:
                                if consumer_node != consumer:
                                    inp_trans_out = consumer_node.name + '_in0'
                                    model.set_tensor_shape(inp_trans_out, (b, c, ho, wo))
                                    dtype = model.get_tensor_datatype(mid_name)
                                    model.set_tensor_datatype(inp_trans_out, dtype)
                                    inp_trans_node = helper.make_node("Transpose", [end_name], [inp_trans_out], perm=[0, 3, 1, 2])
                                    consumer_node.input[0] = inp_trans_out
                                    graph.node.insert(node_ind, inp_trans_node)
                            model.set_tensor_shape(mid_name, (b, hi, wi, c))
                            model.set_tensor_shape(end_name, (b, ho, wo, c))
                            graph.node.remove(consumer)
                            graph.node.insert(node_ind - 1, consumer)
                        else :
                            model.set_tensor_shape(mid_name, (b, hi, wi, c))
                            model.set_tensor_shape(end_name, (b, ho, wo, c))
                            graph.node.remove(consumer)
                            graph.node.insert(node_ind - 1, consumer)
                        graph_modified = True
                elif producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        n.op_type = "MaxPoolNHWC"
                        n.domain = "qonnx.custom_op.general"
                        start_name = producer.input[0]
                        mid_name = n.input[0]
                        end_name = n.output[0]
                        (b, hi, wi, c) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(end_name)
                        producer.input[0] = mid_name                          
                        producer.output[0] = end_name
                        n.input[0] = start_name
                        n.output[0] = mid_name
                        model.set_tensor_shape(mid_name, (b, ho, wo, c))
                        model.set_tensor_shape(end_name, (b, c, ho, wo))
                        graph.node.remove(producer)
                        graph.node.insert(node_ind, producer)
                        graph_modified = True
        return (model, graph_modified)


class InferTriplicateStreamsLayer(Transformation):
    """Insert two DuplicateStreams HLS layer for any tensor with fanout == 3"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            successors = model.find_consumers(node.output[0])
            if successors is not None and len(successors) == 3:
                output_tensor = node.output[0]

                dt = model.get_tensor_datatype(output_tensor)

                # skip conversion for layers with float input
                if not dt.is_integer():
                    continue

                # create clone tensors
                out_shape = model.get_tensor_shape(output_tensor)
                out_tensor_clones = []
                for i in range(4):
                    clone = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                    )
                    model.graph.value_info.append(clone)
                    out_tensor_clones += [clone.name]

                num_ch = int(out_shape[-1])
                vecs = out_shape[:-1]

                # create nodes with no parallelization first
                pe = 1

                dup_node_0 = helper.make_node(
                    "DuplicateStreams_Batch",
                    [output_tensor],
                    out_tensor_clones[0:2],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=dt.name,
                    numInputVectors=vecs,
                    NumOutputStreams=2,
                    outFIFODepths=[2] * 2,
                    name="DuplicateStreams_Batch_" + node.name + "_0",
                )

                dup_node_1 = helper.make_node(
                    "DuplicateStreams_Batch",
                    [out_tensor_clones[0]],
                    out_tensor_clones[2:4],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=dt.name,
                    numInputVectors=vecs,
                    NumOutputStreams=2,
                    outFIFODepths=[2] * 2,
                    name="DuplicateStreams_Batch_" + node.name + "_1",
                )

                graph.node.insert(node_ind, dup_node_0)
                graph.node.insert(node_ind, dup_node_1)

                out_tensors = out_tensor_clones[1:4]

                # connect successors to out tensor clone
                clone_idx = 0
                for successor in successors:
                    for i, succ_input in enumerate(successor.input):
                        if succ_input == output_tensor:
                            successor.input[i] = out_tensors[clone_idx]
                            clone_idx += 1
                            # if one node has multiple connections to the same output
                            # find_direct_successors will return one node per input
                            # so break the inner loop will result in correct behaviour
                            break

                graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
    
class SplitLargeFIFOs(Transformation):
    """Split FIFOs with a depth larger than 32768 into smaller ones
    to ensure that they can be correctly generated."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "StreamingFIFO":
                depth = get_by_name(n.attribute, "depth")
                if depth.i > 32768:
                    n0 = getCustomOp(n)
                    fld_shape = n0.get_folded_output_shape()
                    dtype = n0.get_nodeattr("dataType")
                    impl_style = n0.get_nodeattr("impl_style")
                    ram_style = n0.get_nodeattr("ram_style")
                    shape = model.get_tensor_shape(n.input[0])
                    split_n = math.ceil(depth.i / 32768)
                    fifo_depth = math.ceil(depth.i / split_n)
                    for i in range(split_n):
                        if i == 0:
                            inp = n.input[0]
                        else:
                            inp = n.name + "_" + str(i - 1) + "_out"
                        if i == split_n - 1:
                            outp = n.output[0]
                        else:
                            outp = n.name + "_" + str(i) + "_out"
                            out_tensor = helper.make_tensor_value_info(
                                outp, TensorProto.FLOAT, shape
                            )
                            graph.value_info.append(out_tensor)
                            model.set_tensor_datatype(out_tensor.name, DataType[dtype])
                        fifo_node = helper.make_node(
                            "StreamingFIFO",
                            [inp],
                            [outp],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            depth=fifo_depth,
                            folded_shape=fld_shape,
                            dataType=dtype,
                            impl_style=impl_style,
                            ram_style=ram_style,
                        )
                        graph.node.insert(node_ind + i, fifo_node)

                    graph.node.remove(n)
                    if n.output[0] != "global_out":
                        consumer = model.find_consumer(n.output[0])
                        n1 = getCustomOp(consumer)
                        n1.set_nodeattr("outFIFODepth", fifo_depth)
                    if n.input[0] != "global_in":
                        producer = model.find_producer(n.input[0])
                        n2 = getCustomOp(producer)
                        n2.set_nodeattr("inFIFODepth", fifo_depth)
                    graph_modified = True
        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        return (model, graph_modified)
def step_unet_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run the tidy-up step on given model. This includes shape and datatype
    inference, constant folding, and giving nodes and tensors better names.
    """
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    if VerificationStepType.TIDY_UP_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "initial_python", need_parent=False)

    return model


def step_unet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(MoveLinearPastEltwiseAdd())
    model = model.transform(MoveLinearPastFork())
    model = model.transform(Streamline())
    model = model.transform(MoveIdenticalOpPastJoinOp(['Transpose'], ['Add']))
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model

   
def step_unet_convert_to_hls(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert eligible nodes to `HLSCustomOp` subclasses that represent HLS
    layers. Which nodes and particular configurations can be converted to HLS
    is limited, see the source code of the `convert_to_hls` module for more."""
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())

    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
 
    model = model.transform(to_hls.InferAddStreamsLayer())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(to_hls.InferChannelwiseLinearLayer())
    model = model.transform(CustomMakeMaxPoolNHWC())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(to_hls.InferPool_Batch())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(MakeUpsampleNHWC())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferUpsample())
    model = model.transform(to_hls.InferDuplicateStreamsLayer())
    model = model.transform(InferTriplicateStreamsLayer())
        
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


def step_unet_set_fifo_depths(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Depending on the auto_fifo_depths setting, do one of the following:
    * if auto_fifo_depths=True:  Run the appropriate auto-sizing transformation
    to attempt to determine the FIFO sizes that provide full throughput.
    May take a long time.
    * if auto_fifo_depths=False:  Assume the folding config file contains FIFO
    sizes as well. Runs the `InsertFIFO` transformation, then
    `ApplyConfig(cfg.folding_config_file)`, and finally `RemoveShallowFIFOs`.
    Coherency with config file node naming is ensured by calling
    `GiveUniqueNodeNames`.
    """

    if cfg.auto_fifo_depths:    
        # multi-in/out streams currently not supported in our C++ verilator driver
        model_multi_io = len(model.graph.input) > 1 or len(model.graph.output) > 1
        force_python_sim = model_multi_io or cfg.force_python_rtlsim
        if model_multi_io:
            warnings.warn(
                "Multi-in/out streams currently not supported "
                + "in FINN C++ verilator driver, falling back to Python"
            )
        model = model.transform(
            InsertAndSetFIFODepths(
                cfg._resolve_fpga_part(),
                cfg._resolve_hls_clk_period(),
                vivado_ram_style=cfg.large_fifo_mem_style,
                force_python_sim=force_python_sim,
            )
        )
        # InsertAndSetFIFODepths internally removes any shallow FIFOs
        # so no need to call RemoveShallowFIFOs here
    else:
        # assume folding cfg json contains FIFO sizes too
        # insert DWCs, FIFOs and run ApplyConfig once more
        model = model.transform(InsertDWC())
        # need to make sure all FIFOs are created so that their depth can be
        # set by ApplyConfig, so create_shallow_fifos=True
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        if cfg.folding_config_file is not None:
            model = model.transform(ApplyConfig(cfg.folding_config_file))

    # extract the final configuration and save it as json
    hw_attrs = [
        "PE",
        "SIMD",
        "ram_style",
        "depth",
        "impl_style",
        "resType",
        "mem_mode",
        "runtime_writeable_weights",
        "inFIFODepths",
        "outFIFODepths",
    ]
    extract_model_config_to_json(
        model, cfg.output_dir + "/final_hw_config.json", hw_attrs
    )

    # perform FIFO splitting and shallow FIFO removal only after the final config
    # json file has been written. otherwise, since these transforms may add/remove
    # FIFOs, we get name mismatch problems when trying to reuse the final config.
    
    model = model.transform(SplitLargeFIFOs())
    model = model.transform(RemoveShallowFIFOs())

    # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
    # this will only run for the new nodes (e.g. FIFOs and DWCs)
    model = model.transform(
        PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
    )
    model = model.transform(HLSSynthIP())
    return model
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from custom_steps import *

unet_build_dataflow_steps = [
    "step_qonnx_to_finn",
    step_unet_tidy_up,
    step_unet_streamline,
    step_unet_convert_to_hls,
    "step_create_dataflow_partition",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
    "step_hls_codegen",
    "step_hls_ipgen",
    step_unet_set_fifo_depths,
    "step_synthesize_bitfile",
]

model_file = "../models/resnet18unet_256_256_w4_a4.onnx"
folding_config_file = "resnet18_unet_folding_config.json"

def main():
    cfg = build.DataflowBuildConfig(
    output_dir          = "../outputs",
    synth_clk_period_ns = 5,
    board               = "U250",
    shell_flow_type     = build_cfg.ShellFlowType.VITIS_ALVEO,
    vitis_platform      = "xilinx_u250_gen3x16_xdma_2_1_202010_1",
    steps               = unet_build_dataflow_steps,
    folding_config_file = folding_config_file,
    auto_fifo_depths = False,
    large_fifo_mem_style = build_cfg.LargeFIFOMemStyle.URAM,

    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
    ]
    )
    build.build_dataflow_cfg(model_file, cfg)
    return

if __name__ == '__main__':
    main()

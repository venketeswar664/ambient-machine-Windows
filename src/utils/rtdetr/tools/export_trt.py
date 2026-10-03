#/home/sentinel/Projects/RT-DETR/rtdetrv2_pytorch/tools/export_trt.py
import os
import argparse
import tensorrt as trt

def build_engine(onnx_path, engine_path, min_b, opt_b, max_b, use_fp16=True, verbose=False):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)

    # Build explicit-batch network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"[INFO] Loading ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    # Single builder config
    config = builder.create_builder_config()

    # Workspace (TRT 10+ API; fallback to older API)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    else:
        config.max_workspace_size = 1 << 30

    # FP16 (if supported)
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 enabled")
    elif use_fp16:
        print("[WARN] FP16 requested but not supported; using FP32")

    # Optimization profile
    profile = builder.create_optimization_profile()
    img_name = "images"
    size_name = "orig_target_sizes"

    # Infer input names if they differ
    try:
        num_in = network.num_inputs
    except AttributeError:
        num_in = network.get_nb_inputs()
    for idx in range(num_in):
        inp = network.get_input(idx)
        name = inp.name
        dims = inp.shape
        # A 4D input is likely the image tensor; a 2D(N,2) input is likely sizes
        if len(dims) == 4 and "image" not in name.lower():
            img_name = name
        if len(dims) == 2 and dims[-1] in (2, -1) and "size" not in name.lower():
            size_name = name

    # Assume 3x640x640 static spatial size; change if your model differs
    profile.set_shape(img_name,
                      min=(min_b, 3, 640, 640),
                      opt=(opt_b, 3, 640, 640),
                      max=(max_b, 3, 640, 640))
    profile.set_shape(size_name,
                      min=(1, 2),
                      opt=(opt_b, 2),
                      max=(max_b, 2))
    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine/plan...")
    if hasattr(builder, "build_serialized_network"):  # TRT ≥ 10
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("build_serialized_network() returned None")
        with open(engine_path, "wb") as f:
            f.write(bytearray(plan))
        print(f"[INFO] Saved serialized engine (plan) to: {engine_path}")
    else:  # TRT ≤ 9 compatibility
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Engine build failed")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"[INFO] Saved engine to: {engine_path}")

def main():
    p = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine/Plan")
    p.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model")
    p.add_argument("--saveEngine", "-o", type=str, default="model.engine", help="Path to output .engine/.plan")
    p.add_argument("--maxBatchSize", "-Mb", type=int, default=32)
    p.add_argument("--optBatchSize", "-ob", type=int, default=16)
    p.add_argument("--minBatchSize", "-mb", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="Enable FP16")
    p.add_argument("--verbose", action="store_true", help="Verbose TRT logs")
    args = p.parse_args()

    build_engine(args.onnx, args.saveEngine,
                 args.minBatchSize, args.optBatchSize, args.maxBatchSize,
                 use_fp16=args.fp16, verbose=args.verbose)

if __name__ == "__main__":
    main()
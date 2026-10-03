import tensorrt as trt
from ultralytics import YOLO



def convert_pt_to_engine(model_path):

    # Verify TensorRT installation and version
    print(f"TensorRT Version: {trt.__version__}")
    assert trt.Builder(trt.Logger()), "TensorRT Builder could not be initialized."

    # Initialize YOLO model
    model = YOLO(model_path)

    # Export YOLO model to TensorRT engine file
    print(f"Exporting pt to TensorRT engine: {model_path}")
    try:
        model.export(format='engine')
        print("TensorRT engine export successful.")
    except Exception as e:
        print(f"Failed to export TensorRT engine: {e}")

# run.py

import onnxruntime as ort
import numpy as np

# Print ONNX Runtime version and build info
print(ort.__version__)
print(ort.get_device())

# Load the ONNX model
onnx_model_path = "simplenet_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Function to run inference with ONNX Runtime
def run_onnx_inference(onnx_session, data):
    # Ensure data has the correct batch dimension
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Prepare the input data
    inputs = {onnx_session.get_inputs()[0].name: data.astype(np.float32)}
    
    # Run inference
    output = onnx_session.run(None, inputs)
    
    return output[0]

# Define new data points for testing
new_points = np.array([
    [1.0, 0.5],
    [-1.0, -0.5],
    [2.0, 1.0],
    [-2.0, -1.0]
])

# Test each point separately
for point in new_points:
    # Reshape point to match (1, 2) expected by the model
    point_reshaped = point.reshape(1, 2)
    
    # Run inference using the ONNX model
    predicted_z_onnx = run_onnx_inference(ort_session, point_reshaped)
    
    # Print the results
    print(f"Point {point} -> Predicted z (ONNX) = {predicted_z_onnx[0][0]}")

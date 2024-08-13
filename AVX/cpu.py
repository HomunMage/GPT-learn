import onnxruntime as ort

def get_cpu_features():
    # Check available CPU execution providers
    providers = ort.get_available_providers()
    return providers

features = get_cpu_features()
print("Available CPU features and providers:", features)

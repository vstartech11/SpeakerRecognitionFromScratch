import torch

if torch.cuda.is_available():
    print("CUDA tersedia. Informasi GPU yang terdeteksi:")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_capability = torch.cuda.get_device_capability(i)
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_name}")
        print(f"  - Compute Capability: {gpu_capability}")
        print(f"  - Total Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
        print(f"  - Multiprocessors: {gpu_properties.multi_processor_count}")
else:
    print("CUDA tidak tersedia. Menggunakan CPU.")

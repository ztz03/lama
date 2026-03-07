import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("device0:", torch.cuda.get_device_name(0))

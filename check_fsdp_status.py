"""Quick script to check if FSDP is actually being used."""

import torch
from accelerate import Accelerator, DistributedType
from models.network_25d import create_swin_fir_unet_25d

# Initialize Accelerator with FSDP config
accelerator = Accelerator()

print(f"\n{'='*60}")
print(f"Distributed Type: {accelerator.distributed_type}")
print(f"Num Processes: {accelerator.num_processes}")
print(f"Process Index: {accelerator.process_index}")
print(f"Is Main Process: {accelerator.is_main_process}")

if accelerator.distributed_type == DistributedType.FSDP:
    print(f"✅ FSDP IS ENABLED")
    print(f"FSDP Config: {accelerator.state.fsdp_plugin}")
else:
    print(f"❌ FSDP NOT ENABLED - Using {accelerator.distributed_type}")

print(f"{'='*60}\n")

# Create a small model to test memory
if accelerator.is_main_process:
    print("Creating model...")

model = create_swin_fir_unet_25d(
    in_channels=2,
    out_channels=1,
    num_slices=5,
    base_channels=64,
    channel_mults=(1, 2, 4),
    attention_resolutions=(2, 4),
    num_res_blocks=2,
    radius_emb_dim=256,
    dropout=0.0,
)

if accelerator.is_main_process:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.2f}M")
    print(f"Model size: {total_params * 4 / 1024**3:.2f} GB (fp32)")

# Prepare model
model = accelerator.prepare(model)

if accelerator.is_main_process:
    print(f"\nModel type after prepare: {type(model)}")
    if hasattr(model, '_fsdp_wrapped_module'):
        print("✅ Model is FSDP wrapped")
    elif 'FSDP' in str(type(model)):
        print("✅ Model is FSDP wrapped")
    else:
        print(f"⚠️  Model wrapper: {type(model)}")

# Check memory
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    torch.cuda.empty_cache()
    
    # Create a small test batch
    test_input = torch.randn(1, 2, 5, 256, 256, device=accelerator.device)
    test_cond = torch.randn(1, device=accelerator.device)
    
    with torch.no_grad():
        output = model(test_input[:, :1, :, :, :], test_input[:, 1:, :, :, :], test_cond)
    
    mem_allocated = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
    print(f"\nGPU {accelerator.process_index} Memory after forward pass: {mem_allocated:.2f} GB")

print(f"\n{'='*60}\n")

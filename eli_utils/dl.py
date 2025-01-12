import gc

def clear_vram():
    import torch
  
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

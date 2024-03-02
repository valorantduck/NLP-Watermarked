import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import numpy as np
import shutil
from tqdm import tqdm
from math import sqrt


rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)  # fib47 is prime
table_size = 1_000_003
fixed_table = torch.randperm(
    1_000_003, device=torch.device("cpu"), generator=rng
)  # actually faster than I thought


def hashint(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % table_size] + 1
    )  # minor cheat here, this function always return CPU values


def detect_watermark(input_ids, gamma=0.5, codebook_size=16384, z_threshold=4.0, watermark_type="simple"):
    """
    水印检测算法
    """
    
    g_cuda = torch.Generator(device=input_ids.device)  # 用于设置随机数
    large_prime = 15485863
    green_token_count = 0
    if watermark_type == "simple":
        for l_idx in range(1, len(input_ids)-1):    # 遍历每一个索引进行加密，除了第一个和最后一个作为context使用
            curr_token = input_ids[l_idx]
            g_cuda.manual_seed(large_prime*input_ids[l_idx-1].item())
            green_size = int(codebook_size * gamma)
            green_ids = torch.randperm(codebook_size, 
                                    device=input_ids.device, 
                                    generator=g_cuda)[(codebook_size - green_size) :]    # 通过R表来得到G表
            if curr_token in green_ids:
                green_token_count += 1
        
        numer = green_token_count - gamma * (len(input_ids)-2)      # 不计算第0个和最后一个
        denom = sqrt((len(input_ids)-2) * gamma * (1 - gamma))
        z = numer / denom
    
    else:
        green_ids_all = []
        for l_idx in range(1, len(input_ids)-1):    # 遍历每一个索引进行加密，除了第一个和最后一个作为context使用
            curr_token = input_ids[l_idx]
            context = input_ids[l_idx-1: l_idx+2]
            if watermark_type == "additive":
                g_cuda.manual_seed(large_prime*context.sum().item())
            elif watermark_type == "min":
                g_cuda.manual_seed(large_prime*context.min().item())
            elif watermark_type == "skip":
                skip_hash = hashint(large_prime * context[0]).item()
                g_cuda.manual_seed(large_prime*skip_hash)
                
            green_size = int(codebook_size * gamma)
            green_ids = torch.randperm(codebook_size, 
                                    device=input_ids.device, 
                                    generator=g_cuda)[(codebook_size - green_size) :]    # 通过R表来得到G表
            if curr_token in green_ids:
                green_ids_all.append(1)
            else:
                green_ids_all.append(0)
        
        green_ids_all = torch.tensor(green_ids_all).to(input_ids.device)
        len_full_context = len(green_ids_all)
        sizes = range(1, len_full_context)
        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        partial_sum_id_table = torch.cumsum(green_ids_all, dim=0)
        
        for idx, size in enumerate(sizes):      # 遍历长度T
            if size <= len_full_context:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                window_score[0] = partial_sum_id_table[size - 1]    # p_i
                window_score[1:] = partial_sum_id_table[size::1] - partial_sum_id_table[:-size:1]
                batched_z_score_enum = window_score - gamma * size
                z_score_denom = sqrt(size * gamma * (1 - gamma))
                batched_z_score = batched_z_score_enum / z_score_denom
                
                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score
                
                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::1] = torch.maximum(
                    cumulative_eff_z_score[size::1], z_score_at_effective_T[:-1]
                )
        
        z, _ = z_score_max_per_window.max(dim=0)

    predict = z > z_threshold
    
    return predict, z

        
        
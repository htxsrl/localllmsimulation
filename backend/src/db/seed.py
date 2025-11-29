"""Database seed data from real benchmarks."""

from .connection import get_cursor, init_database


def seed_hardware():
    """Insert hardware data with real specs from Apple and benchmarks."""
    hardware_data = [
        # ==================== APPLE SILICON ====================
        # Organized by: Product Type > Chip Generation > RAM Size

        # --- MacBook Air ---
        # M1
        ("MacBook Air M1 8GB", "Apple", "M1", 8, 68, "unified", 8, "llama.cpp#4167"),
        # M2
        ("MacBook Air M2 8GB", "Apple", "M2", 8, 100, "unified", 10, "llama.cpp#4167"),
        ("MacBook Air M2 16GB", "Apple", "M2", 16, 100, "unified", 10, "llama.cpp#4167"),
        ("MacBook Air M2 24GB", "Apple", "M2", 24, 100, "unified", 10, "llama.cpp#4167"),
        # M3
        ("MacBook Air M3 8GB", "Apple", "M3", 8, 100, "unified", 10, "llama.cpp#4167"),
        ("MacBook Air M3 16GB", "Apple", "M3", 16, 100, "unified", 10, "llama.cpp#4167"),
        ("MacBook Air M3 24GB", "Apple", "M3", 24, 100, "unified", 10, "llama.cpp#4167"),
        # M4
        ("MacBook Air M4 16GB", "Apple", "M4", 16, 120, "unified", 10, "apple.com"),
        ("MacBook Air M4 24GB", "Apple", "M4", 24, 120, "unified", 10, "apple.com"),
        ("MacBook Air M4 32GB", "Apple", "M4", 32, 120, "unified", 10, "apple.com"),

        # --- MacBook Pro ---
        # M1
        ("MacBook Pro M1 16GB", "Apple", "M1", 16, 68, "unified", 8, "llama.cpp#4167"),
        # M1 Pro
        ("MacBook Pro M1 Pro 16GB", "Apple", "M1 Pro", 16, 200, "unified", 16, "llama.cpp#4167"),
        ("MacBook Pro M1 Pro 32GB", "Apple", "M1 Pro", 32, 200, "unified", 16, "llama.cpp#4167"),
        # M1 Max
        ("MacBook Pro M1 Max 32GB", "Apple", "M1 Max", 32, 400, "unified", 32, "llama.cpp#4167"),
        ("MacBook Pro M1 Max 64GB", "Apple", "M1 Max", 64, 400, "unified", 32, "llama.cpp#4167"),
        # M2 Pro
        ("MacBook Pro M2 Pro 16GB", "Apple", "M2 Pro", 16, 200, "unified", 19, "llama.cpp#4167"),
        ("MacBook Pro M2 Pro 32GB", "Apple", "M2 Pro", 32, 200, "unified", 19, "llama.cpp#4167"),
        # M2 Max
        ("MacBook Pro M2 Max 32GB", "Apple", "M2 Max", 32, 400, "unified", 38, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 64GB", "Apple", "M2 Max", 64, 400, "unified", 38, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 96GB", "Apple", "M2 Max", 96, 400, "unified", 38, "llama.cpp#4167"),
        # M3
        ("MacBook Pro M3 16GB", "Apple", "M3", 16, 100, "unified", 10, "llama.cpp#4167"),
        ("MacBook Pro M3 24GB", "Apple", "M3", 24, 100, "unified", 10, "llama.cpp#4167"),
        # M3 Pro
        ("MacBook Pro M3 Pro 18GB", "Apple", "M3 Pro", 18, 150, "unified", 18, "llama.cpp#4167"),
        ("MacBook Pro M3 Pro 36GB", "Apple", "M3 Pro", 36, 150, "unified", 18, "llama.cpp#4167"),
        # M3 Max
        ("MacBook Pro M3 Max 36GB", "Apple", "M3 Max", 36, 400, "unified", 40, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 48GB", "Apple", "M3 Max", 48, 400, "unified", 40, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Apple", "M3 Max", 64, 400, "unified", 40, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 128GB", "Apple", "M3 Max", 128, 400, "unified", 40, "llama.cpp#4167"),
        # M4
        ("MacBook Pro M4 16GB", "Apple", "M4", 16, 120, "unified", 10, "apple.com"),
        ("MacBook Pro M4 24GB", "Apple", "M4", 24, 120, "unified", 10, "apple.com"),
        ("MacBook Pro M4 32GB", "Apple", "M4", 32, 120, "unified", 10, "apple.com"),
        # M4 Pro
        ("MacBook Pro M4 Pro 24GB", "Apple", "M4 Pro", 24, 273, "unified", 20, "apple.com"),
        ("MacBook Pro M4 Pro 48GB", "Apple", "M4 Pro", 48, 273, "unified", 20, "apple.com"),
        # M4 Max
        ("MacBook Pro M4 Max 36GB", "Apple", "M4 Max", 36, 546, "unified", 40, "apple.com"),
        ("MacBook Pro M4 Max 48GB", "Apple", "M4 Max", 48, 546, "unified", 40, "apple.com"),
        ("MacBook Pro M4 Max 64GB", "Apple", "M4 Max", 64, 546, "unified", 40, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 128GB", "Apple", "M4 Max", 128, 546, "unified", 40, "llama.cpp#4167"),
        # M5
        ("MacBook Pro M5 16GB", "Apple", "M5", 16, 154, "unified", 10, "apple.com"),
        ("MacBook Pro M5 24GB", "Apple", "M5", 24, 154, "unified", 10, "apple.com"),
        ("MacBook Pro M5 32GB", "Apple", "M5", 32, 154, "unified", 10, "apple.com"),

        # --- Mac Mini ---
        # M2
        ("Mac Mini M2 16GB", "Apple", "M2", 16, 100, "unified", 10, "llama.cpp#4167"),
        ("Mac Mini M2 24GB", "Apple", "M2", 24, 100, "unified", 10, "llama.cpp#4167"),
        # M4
        ("Mac Mini M4 16GB", "Apple", "M4", 16, 120, "unified", 10, "apple.com"),
        ("Mac Mini M4 24GB", "Apple", "M4", 24, 120, "unified", 10, "apple.com"),
        ("Mac Mini M4 32GB", "Apple", "M4", 32, 120, "unified", 10, "apple.com"),
        # M4 Pro
        ("Mac Mini M4 Pro 24GB", "Apple", "M4 Pro", 24, 273, "unified", 20, "apple.com"),
        ("Mac Mini M4 Pro 48GB", "Apple", "M4 Pro", 48, 273, "unified", 20, "apple.com"),
        ("Mac Mini M4 Pro 64GB", "Apple", "M4 Pro", 64, 273, "unified", 20, "apple.com"),

        # --- Mac Studio ---
        # M1 Ultra
        ("Mac Studio M1 Ultra 64GB", "Apple", "M1 Ultra", 64, 800, "unified", 64, "llama.cpp#4167"),
        ("Mac Studio M1 Ultra 128GB", "Apple", "M1 Ultra", 128, 800, "unified", 64, "llama.cpp#4167"),
        # M2 Max
        ("Mac Studio M2 Max 32GB", "Apple", "M2 Max", 32, 400, "unified", 38, "llama.cpp#4167"),
        ("Mac Studio M2 Max 64GB", "Apple", "M2 Max", 64, 400, "unified", 38, "llama.cpp#4167"),
        ("Mac Studio M2 Max 96GB", "Apple", "M2 Max", 96, 400, "unified", 38, "llama.cpp#4167"),
        # M2 Ultra
        ("Mac Studio M2 Ultra 64GB", "Apple", "M2 Ultra", 64, 800, "unified", 76, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Apple", "M2 Ultra", 128, 800, "unified", 76, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 192GB", "Apple", "M2 Ultra", 192, 800, "unified", 76, "llama.cpp#4167"),
        # M3 Max
        ("Mac Studio M3 Max 36GB", "Apple", "M3 Max", 36, 400, "unified", 40, "apple.com"),
        ("Mac Studio M3 Max 64GB", "Apple", "M3 Max", 64, 400, "unified", 40, "apple.com"),
        ("Mac Studio M3 Max 128GB", "Apple", "M3 Max", 128, 400, "unified", 40, "apple.com"),
        # M3 Ultra
        ("Mac Studio M3 Ultra 64GB", "Apple", "M3 Ultra", 64, 800, "unified", 80, "apple.com"),
        ("Mac Studio M3 Ultra 128GB", "Apple", "M3 Ultra", 128, 800, "unified", 80, "apple.com"),
        ("Mac Studio M3 Ultra 256GB", "Apple", "M3 Ultra", 256, 800, "unified", 80, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 512GB", "Apple", "M3 Ultra", 512, 800, "unified", 80, "apple.com"),
        # M4 Max
        ("Mac Studio M4 Max 36GB", "Apple", "M4 Max", 36, 546, "unified", 40, "apple.com"),
        ("Mac Studio M4 Max 64GB", "Apple", "M4 Max", 64, 546, "unified", 40, "apple.com"),
        ("Mac Studio M4 Max 128GB", "Apple", "M4 Max", 128, 546, "unified", 40, "apple.com"),

        # --- Mac Pro ---
        # M2 Ultra
        ("Mac Pro M2 Ultra 192GB", "Apple", "M2 Ultra", 192, 800, "unified", 76, "llama.cpp#4167"),

        # ==================== NVIDIA ====================

        # NVIDIA RTX 30 Series
        ("RTX 3060 12GB", "NVIDIA", "RTX 3060", 12, 360, "vram", 3584, "techpowerup.com"),
        ("RTX 3060 Ti 8GB", "NVIDIA", "RTX 3060 Ti", 8, 448, "vram", 4864, "techpowerup.com"),
        ("RTX 3070 8GB", "NVIDIA", "RTX 3070", 8, 448, "vram", 5888, "techpowerup.com"),
        ("RTX 3070 Ti 8GB", "NVIDIA", "RTX 3070 Ti", 8, 608, "vram", 6144, "techpowerup.com"),
        ("RTX 3080 10GB", "NVIDIA", "RTX 3080", 10, 760, "vram", 8704, "techpowerup.com"),
        ("RTX 3080 12GB", "NVIDIA", "RTX 3080", 12, 912, "vram", 8960, "techpowerup.com"),
        ("RTX 3080 Ti 12GB", "NVIDIA", "RTX 3080 Ti", 12, 912, "vram", 10240, "techpowerup.com"),
        ("RTX 3090 24GB", "NVIDIA", "RTX 3090", 24, 936, "vram", 10496, "techpowerup.com"),
        ("RTX 3090 Ti 24GB", "NVIDIA", "RTX 3090 Ti", 24, 1008, "vram", 10752, "techpowerup.com"),

        # NVIDIA RTX 40 Series
        ("RTX 4060 8GB", "NVIDIA", "RTX 4060", 8, 272, "vram", 3072, "techpowerup.com"),
        ("RTX 4060 Ti 8GB", "NVIDIA", "RTX 4060 Ti", 8, 288, "vram", 4352, "techpowerup.com"),
        ("RTX 4060 Ti 16GB", "NVIDIA", "RTX 4060 Ti", 16, 288, "vram", 4352, "techpowerup.com"),
        ("RTX 4070 12GB", "NVIDIA", "RTX 4070", 12, 504, "vram", 5888, "techpowerup.com"),
        ("RTX 4070 Super 12GB", "NVIDIA", "RTX 4070 Super", 12, 504, "vram", 7168, "techpowerup.com"),
        ("RTX 4070 Ti 12GB", "NVIDIA", "RTX 4070 Ti", 12, 504, "vram", 7680, "techpowerup.com"),
        ("RTX 4070 Ti Super 16GB", "NVIDIA", "RTX 4070 Ti Super", 16, 672, "vram", 8448, "techpowerup.com"),
        ("RTX 4080 16GB", "NVIDIA", "RTX 4080", 16, 717, "vram", 9728, "techpowerup.com"),
        ("RTX 4080 Super 16GB", "NVIDIA", "RTX 4080 Super", 16, 736, "vram", 10240, "techpowerup.com"),
        ("RTX 4090 24GB", "NVIDIA", "RTX 4090", 24, 1008, "vram", 16384, "techpowerup.com"),
        ("RTX 4090 Laptop 16GB", "NVIDIA", "RTX 4090 Laptop", 16, 576, "vram", 9728, "techpowerup.com"),
        ("2x RTX 4090 48GB", "NVIDIA", "2x RTX 4090", 48, 2016, "vram", 32768, "techpowerup.com"),

        # NVIDIA RTX 50 Series
        ("RTX 5090 32GB", "NVIDIA", "RTX 5090", 32, 1792, "vram", 21760, "nvidia.com"),
        ("RTX 5090 Laptop 24GB", "NVIDIA", "RTX 5090 Laptop", 24, 896, "vram", 10496, "nvidia.com"),
        ("RTX 5080 16GB", "NVIDIA", "RTX 5080", 16, 960, "vram", 10752, "nvidia.com"),
        ("RTX 5070 Ti 16GB", "NVIDIA", "RTX 5070 Ti", 16, 896, "vram", 8960, "nvidia.com"),
        ("RTX 5060 Ti 16GB", "NVIDIA", "RTX 5060 Ti", 16, 448, "vram", 4608, "nvidia.com"),
        ("RTX 5070 12GB", "NVIDIA", "RTX 5070", 12, 672, "vram", 6144, "nvidia.com"),
        ("2x RTX 5090 64GB", "NVIDIA", "2x RTX 5090", 64, 3584, "vram", 43520, "nvidia.com"),

        # NVIDIA Professional / Workstation
        ("RTX A4000 16GB", "NVIDIA", "RTX A4000", 16, 448, "vram", 6144, "techpowerup.com"),
        ("RTX A5000 24GB", "NVIDIA", "RTX A5000", 24, 768, "vram", 8192, "techpowerup.com"),
        ("RTX A6000 48GB", "NVIDIA", "RTX A6000", 48, 768, "vram", 10752, "techpowerup.com"),
        ("L40S 48GB", "NVIDIA", "L40S", 48, 864, "vram", 18176, "nvidia.com"),

        # NVIDIA Data Center - A100
        ("A100 40GB PCIe", "NVIDIA", "A100", 40, 1555, "vram", 6912, "nvidia.com"),
        ("A100 80GB PCIe", "NVIDIA", "A100", 80, 2039, "vram", 6912, "nvidia.com"),
        ("A100 80GB SXM", "NVIDIA", "A100 SXM", 80, 2039, "vram", 6912, "nvidia.com"),
        ("2x A100 80GB", "NVIDIA", "2x A100", 160, 4078, "vram", 13824, "nvidia.com"),
        ("4x A100 80GB", "NVIDIA", "4x A100", 320, 8156, "vram", 27648, "nvidia.com"),
        ("8x A100 80GB (DGX A100)", "NVIDIA", "DGX A100", 640, 16312, "vram", 55296, "nvidia.com"),

        # NVIDIA Data Center - H100
        ("H100 80GB PCIe", "NVIDIA", "H100 PCIe", 80, 2039, "vram", 14592, "nvidia.com"),
        ("H100 80GB SXM", "NVIDIA", "H100 SXM", 80, 3350, "vram", 16896, "nvidia.com"),
        ("2x H100 80GB", "NVIDIA", "2x H100", 160, 6700, "vram", 33792, "nvidia.com"),
        ("4x H100 80GB", "NVIDIA", "4x H100", 320, 13400, "vram", 67584, "nvidia.com"),
        ("8x H100 80GB (DGX H100)", "NVIDIA", "DGX H100", 640, 26800, "vram", 135168, "nvidia.com"),

        # NVIDIA Data Center - H200
        ("H200 141GB SXM", "NVIDIA", "H200", 141, 4800, "vram", 16896, "nvidia.com"),
        ("8x H200 (DGX H200)", "NVIDIA", "DGX H200", 1128, 38400, "vram", 135168, "nvidia.com"),

        # NVIDIA DGX Spark (GB10 Grace Blackwell)
        ("DGX Spark 128GB", "NVIDIA", "GB10", 128, 273, "unified", 6144, "nvidia.com"),
        ("2x DGX Spark 256GB", "NVIDIA", "2x GB10", 256, 546, "unified", 12288, "nvidia.com"),

        # AMD GPUs
        ("RX 7600 8GB", "AMD", "RX 7600", 8, 288, "vram", 2048, "techpowerup.com"),
        ("RX 7700 XT 12GB", "AMD", "RX 7700 XT", 12, 432, "vram", 3456, "techpowerup.com"),
        ("RX 7800 XT 16GB", "AMD", "RX 7800 XT", 16, 624, "vram", 3840, "techpowerup.com"),
        ("RX 7900 XT 20GB", "AMD", "RX 7900 XT", 20, 800, "vram", 5376, "techpowerup.com"),
        ("RX 7900 XTX 24GB", "AMD", "RX 7900 XTX", 24, 960, "vram", 6144, "techpowerup.com"),
        ("RX 7900 GRE 16GB", "AMD", "RX 7900 GRE", 16, 576, "vram", 5120, "techpowerup.com"),
    ]

    with get_cursor() as cursor:
        cursor.executemany(
            """
            INSERT OR IGNORE INTO hardware
            (name, vendor, chip, memory_gb, memory_bandwidth_gbs, memory_type, gpu_cores, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            hardware_data,
        )


def seed_models():
    """Insert model data."""
    models_data = [
        # Llama 3.1 series
        ("Llama 3.1", None, 8, "dense", 131072, "meta.com"),
        ("Llama 3.1", None, 70, "dense", 131072, "meta.com"),
        ("Llama 3.1", None, 405, "dense", 131072, "meta.com"),
        # Llama 3.2 series
        ("Llama 3.2", None, 1, "dense", 131072, "meta.com"),
        ("Llama 3.2", None, 3, "dense", 131072, "meta.com"),
        ("Llama 3.2", "Vision", 11, "dense", 131072, "meta.com"),
        ("Llama 3.2", "Vision", 90, "dense", 131072, "meta.com"),
        # Llama 3.3
        ("Llama 3.3", None, 70, "dense", 131072, "meta.com"),

        # Qwen 2.5 series
        ("Qwen 2.5", None, 0.5, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", None, 1.5, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", None, 3, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", None, 7, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", None, 14, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", None, 32, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", None, 72, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", "Coder", 7, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", "Coder", 14, "dense", 131072, "qwen.ai"),
        ("Qwen 2.5", "Coder", 32, "dense", 131072, "qwen.ai"),
        ("QwQ", None, 32, "dense", 131072, "qwen.ai"),

        # Mistral/Mixtral
        ("Mistral", "7B v0.3", 7, "dense", 32768, "mistral.ai"),
        ("Mistral", "Nemo", 12, "dense", 128000, "mistral.ai"),
        ("Mistral", "Small 3.1", 24, "dense", 128000, "mistral.ai"),
        ("Mistral", "Large 2", 123, "dense", 128000, "mistral.ai"),
        ("Mixtral", "8x7B", 46.7, "moe", 32768, "mistral.ai"),
        ("Mixtral", "8x22B", 141, "moe", 65536, "mistral.ai"),
        ("Codestral", None, 22, "dense", 32768, "mistral.ai"),
        ("Pixtral", "12B", 12, "dense", 128000, "mistral.ai"),

        # Phi series (Microsoft)
        ("Phi-3", "Mini", 3.8, "dense", 128000, "microsoft.com"),
        ("Phi-3", "Small", 7, "dense", 128000, "microsoft.com"),
        ("Phi-3", "Medium", 14, "dense", 128000, "microsoft.com"),
        ("Phi-3.5", "Mini", 3.8, "dense", 128000, "microsoft.com"),
        ("Phi-3.5", "MoE", 41.9, "moe", 128000, "microsoft.com"),
        ("Phi-4", None, 14, "dense", 16384, "microsoft.com"),

        # Gemma 2 (Google)
        ("Gemma 2", None, 2, "dense", 8192, "google.com"),
        ("Gemma 2", None, 9, "dense", 8192, "google.com"),
        ("Gemma 2", None, 27, "dense", 8192, "google.com"),
        # Gemma 3 (Google)
        ("Gemma 3", None, 4, "dense", 128000, "google.com"),
        ("Gemma 3", None, 12, "dense", 128000, "google.com"),
        ("Gemma 3", None, 27, "dense", 128000, "google.com"),

        # DeepSeek
        ("DeepSeek", "Coder V2", 16, "dense", 128000, "deepseek.com"),
        ("DeepSeek", "V2.5", 236, "moe", 128000, "deepseek.com"),
        ("DeepSeek", "V3", 671, "moe", 128000, "deepseek.com"),
        ("DeepSeek", "R1-Lite", 52, "moe", 128000, "deepseek.com"),
        ("DeepSeek", "R1 Distill Qwen", 1.5, "dense", 128000, "deepseek.com"),
        ("DeepSeek", "R1 Distill Qwen", 7, "dense", 128000, "deepseek.com"),
        ("DeepSeek", "R1 Distill Qwen", 14, "dense", 128000, "deepseek.com"),
        ("DeepSeek", "R1 Distill Qwen", 32, "dense", 128000, "deepseek.com"),
        ("DeepSeek", "R1 Distill Llama", 8, "dense", 128000, "deepseek.com"),
        ("DeepSeek", "R1 Distill Llama", 70, "dense", 128000, "deepseek.com"),

        # Command R (Cohere)
        ("Command R", None, 35, "dense", 128000, "cohere.com"),
        ("Command R+", None, 104, "dense", 128000, "cohere.com"),
        ("Aya", "Expanse", 8, "dense", 8192, "cohere.com"),
        ("Aya", "Expanse", 32, "dense", 8192, "cohere.com"),

        # OLMo (AI2)
        ("OLMo 2", None, 7, "dense", 4096, "allenai.org"),
        ("OLMo 2", None, 13, "dense", 4096, "allenai.org"),

        # Falcon 3 (TII)
        ("Falcon 3", None, 1, "dense", 8192, "tii.ae"),
        ("Falcon 3", None, 3, "dense", 8192, "tii.ae"),
        ("Falcon 3", None, 7, "dense", 8192, "tii.ae"),
        ("Falcon 3", None, 10, "dense", 8192, "tii.ae"),

        # SmolLM2 (HuggingFace)
        ("SmolLM2", None, 0.135, "dense", 8192, "huggingface.co"),
        ("SmolLM2", None, 0.36, "dense", 8192, "huggingface.co"),
        ("SmolLM2", None, 1.7, "dense", 8192, "huggingface.co"),

        # Granite 3.1 (IBM)
        ("Granite 3.1", None, 2, "dense", 128000, "ibm.com"),
        ("Granite 3.1", None, 8, "dense", 128000, "ibm.com"),
        ("Granite 3.1", "MoE", 3, "moe", 128000, "ibm.com"),

        # Yi 1.5 (01.AI)
        ("Yi 1.5", None, 6, "dense", 4096, "01.ai"),
        ("Yi 1.5", None, 9, "dense", 4096, "01.ai"),
        ("Yi 1.5", None, 34, "dense", 4096, "01.ai"),

        # InternLM 2.5 (Shanghai AI Lab)
        ("InternLM 2.5", None, 7, "dense", 32768, "internlm.org"),
        ("InternLM 2.5", None, 20, "dense", 32768, "internlm.org"),

        # Nemotron (NVIDIA)
        ("Nemotron", None, 8, "dense", 8192, "nvidia.com"),
        ("Nemotron", None, 70, "dense", 8192, "nvidia.com"),

        # StarCoder 2 (BigCode)
        ("StarCoder 2", None, 3, "dense", 16384, "bigcode.co"),
        ("StarCoder 2", None, 7, "dense", 16384, "bigcode.co"),
        ("StarCoder 2", None, 15, "dense", 16384, "bigcode.co"),

        # GLM-4 (Zhipu AI)
        ("GLM-4", None, 9, "dense", 128000, "zhipuai.cn"),

        # Nous Research
        ("Hermes 3", None, 8, "dense", 131072, "nousresearch.com"),
        ("Hermes 3", None, 70, "dense", 131072, "nousresearch.com"),
        ("Hermes 3", None, 405, "dense", 131072, "nousresearch.com"),
    ]

    with get_cursor() as cursor:
        cursor.executemany(
            """
            INSERT OR IGNORE INTO models
            (family, variant, size_b, architecture, context_default, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            models_data,
        )


def seed_quantizations():
    """Insert quantization data for models."""
    quant_configs = [
        ("F16", 16.0, 2.0),
        ("Q8_0", 8.5, 1.0625),
        ("Q6_K", 6.5, 0.8125),
        ("Q5_K_M", 5.5, 0.6875),
        ("Q4_K_M", 4.5, 0.5625),
        ("Q4_0", 4.0, 0.5),
        ("Q3_K_M", 3.5, 0.4375),
        ("Q2_K", 2.5, 0.3125),
        ("IQ4_XS", 4.25, 0.53),
        ("IQ3_XS", 3.3, 0.41),
        ("IQ2_XS", 2.3, 0.29),
    ]

    with get_cursor() as cursor:
        cursor.execute("SELECT id, size_b FROM models")
        models = cursor.fetchall()

        for model_id, size_b in models:
            for quant_type, bits_per_weight, multiplier in quant_configs:
                size_gb = round(size_b * multiplier, 2)
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO quantizations
                    (model_id, quant_type, size_gb, bits_per_weight)
                    VALUES (?, ?, ?, ?)
                    """,
                    (model_id, quant_type, size_gb, bits_per_weight),
                )


def seed_benchmarks():
    """Insert real benchmark data from various sources with context length variations."""

    # Source URLs for reference
    SOURCES = {
        "llama.cpp#4167": "https://github.com/ggml-org/llama.cpp/discussions/4167",
        "llama.cpp#10879": "https://github.com/ggml-org/llama.cpp/discussions/10879",
        "hardware-corner": "https://www.hardware-corner.net/rtx-4090-llm-benchmarks/",
        "hardware-corner-5090": "https://www.hardware-corner.net/rtx-5090-llm-benchmarks/",
        "gpu-bench-repo": "https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference",
        "deepnewz": "https://deepnewz.com/ai-modeling/llama-3-3-70b-model-achieves-10-tokens-per-second-on-64gb-m3-max-12-tokens-per-980c01a7",
        "macrumors": "https://forums.macrumors.com/threads/m4-max-studio-128gb-llm-testing.2453816/",
        "valdi-h100": "https://docs.valdi.ai/llms/performance/gpu/H100/llama3.1-inference-testing/",
        "ori-benchmark": "https://www.ori.co/blog/benchmarking-llama-3.1-8b-instruct-on-nvidia-h100-and-a100-chips-with-the-vllm-inferencing-engine",
        "databasemart-a100": "https://www.databasemart.com/blog/ollama-gpu-benchmark-a100-40gb",
        "databasemart-5090": "https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx5090",
        "runpod-5090": "https://www.runpod.io/blog/rtx-5090-llm-benchmarks",
        "lmsys-dgx-spark": "https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/",
        "nvidia-h100-blog": "https://developer.nvidia.com/blog/achieving-top-inference-performance-with-the-nvidia-h100-tensor-core-gpu-and-nvidia-tensorrt-llm/",
        "localscore": "https://www.localscore.ai/",
        "localscore-github": "https://github.com/cjpais/LocalScore",
    }

    # Format: (hw_name, model_family, model_size, quant_type, context_length, gen_tps, prompt_tps, source, ttft_ms)
    # ttft_ms is optional (9th element) - only LocalScore benchmarks have measured TTFT
    benchmarks_data = [
        # ========== RTX 4090 - Different Context Lengths (from hardware-corner.net) ==========
        # 8B Q4_K_M at various context lengths
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 4096, 131.0, 9121.0, "hardware-corner"),
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 8192, 119.4, 7907.0, "hardware-corner"),
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 16384, 96.1, 5697.0, "hardware-corner"),
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 32768, 77.4, 4224.0, "hardware-corner"),
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 65536, 53.1, 2614.0, "hardware-corner"),
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 131072, 32.3, 1451.0, "hardware-corner"),

        # 14B Q4_K_M at various context lengths
        ("RTX 4090 24GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 82.8, 5358.0, "hardware-corner"),
        ("RTX 4090 24GB", "Qwen 2.5", 14, "Q4_K_M", 8192, 77.5, 4621.0, "hardware-corner"),
        ("RTX 4090 24GB", "Qwen 2.5", 14, "Q4_K_M", 16384, 63.2, 3355.0, "hardware-corner"),
        ("RTX 4090 24GB", "Qwen 2.5", 14, "Q4_K_M", 32768, 55.5, 2511.0, "hardware-corner"),
        ("RTX 4090 24GB", "Qwen 2.5", 14, "Q4_K_M", 65536, 38.7, 1545.0, "hardware-corner"),

        # 32B Q4_K_M at various context lengths
        ("RTX 4090 24GB", "Qwen 2.5", 32, "Q4_K_M", 4096, 38.9, 2393.0, "hardware-corner"),
        ("RTX 4090 24GB", "Qwen 2.5", 32, "Q4_K_M", 8192, 36.9, 2093.0, "hardware-corner"),
        ("RTX 4090 24GB", "Qwen 2.5", 32, "Q4_K_M", 16384, 33.8, 1685.0, "hardware-corner"),

        # MoE 30B (Mixtral-like) at various context lengths
        ("RTX 4090 24GB", "Mixtral", 46.7, "Q4_K_M", 4096, 195.8, 6266.0, "hardware-corner"),
        ("RTX 4090 24GB", "Mixtral", 46.7, "Q4_K_M", 8192, 171.9, 5415.0, "hardware-corner"),
        ("RTX 4090 24GB", "Mixtral", 46.7, "Q4_K_M", 16384, 130.1, 3802.0, "hardware-corner"),
        ("RTX 4090 24GB", "Mixtral", 46.7, "Q4_K_M", 32768, 102.6, 2726.0, "hardware-corner"),
        ("RTX 4090 24GB", "Mixtral", 46.7, "Q4_K_M", 57344, 74.6, 1718.0, "hardware-corner"),

        # ========== RTX 4090 - From GPU benchmark repo ==========
        ("RTX 4090 24GB", "Llama 3.1", 8, "Q4_K_M", 2048, 127.7, 9500.0, "gpu-bench-repo"),
        ("RTX 4090 24GB", "Llama 3.1", 8, "F16", 2048, 54.3, 5200.0, "gpu-bench-repo"),

        # ========== RTX 3090 - Context scaling ==========
        ("RTX 3090 24GB", "Llama 3.1", 8, "Q4_K_M", 2048, 111.7, 4500.0, "gpu-bench-repo"),
        ("RTX 3090 24GB", "Llama 3.1", 8, "Q4_K_M", 4096, 105.0, 4000.0, "llama.cpp#10879"),
        ("RTX 3090 24GB", "Llama 3.1", 8, "Q4_K_M", 8192, 92.0, 3500.0, "llama.cpp#10879"),
        ("RTX 3090 24GB", "Llama 3.1", 8, "Q4_K_M", 16384, 75.0, 2800.0, "llama.cpp#10879"),
        ("RTX 3090 24GB", "Llama 3.1", 8, "Q4_K_M", 32768, 58.0, 2000.0, "llama.cpp#10879"),

        # ========== RTX 4080 - Context scaling ==========
        ("RTX 4080 16GB", "Llama 3.1", 8, "Q4_K_M", 2048, 98.0, 4200.0, "llama.cpp#10879"),
        ("RTX 4080 16GB", "Llama 3.1", 8, "Q4_K_M", 4096, 92.0, 3800.0, "llama.cpp#10879"),
        ("RTX 4080 16GB", "Llama 3.1", 8, "Q4_K_M", 8192, 82.0, 3200.0, "llama.cpp#10879"),
        ("RTX 4080 16GB", "Llama 3.1", 8, "Q4_K_M", 16384, 65.0, 2400.0, "llama.cpp#10879"),

        # ========== Apple Silicon - M4 Max with context scaling ==========
        ("MacBook Pro M4 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 2048, 83.0, 885.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 4096, 80.0, 820.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 8192, 75.0, 720.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 16384, 68.0, 580.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 32768, 58.0, 420.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.1", 8, "F16", 2048, 31.6, 922.8, "llama.cpp#4167"),

        # M4 Max with 70B model
        ("MacBook Pro M4 Max 128GB", "Llama 3.1", 70, "Q4_K_M", 2048, 12.0, 45.0, "deepnewz"),
        ("MacBook Pro M4 Max 128GB", "Llama 3.1", 70, "Q4_K_M", 4096, 11.5, 42.0, "macrumors"),
        ("MacBook Pro M4 Max 128GB", "Llama 3.1", 70, "Q4_K_M", 8192, 10.8, 38.0, "macrumors"),
        ("MacBook Pro M4 Max 128GB", "Llama 3.1", 70, "Q4_K_M", 16384, 9.5, 32.0, "macrumors"),
        ("MacBook Pro M4 Max 128GB", "Llama 3.3", 70, "Q4_K_M", 2048, 12.0, 45.0, "deepnewz"),

        # ========== Apple Silicon - M3 Max with context scaling ==========
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 2048, 66.3, 759.7, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 4096, 63.0, 700.0, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 8192, 58.0, 600.0, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 16384, 50.0, 480.0, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 32768, 40.0, 350.0, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 8, "F16", 2048, 25.1, 779.2, "llama.cpp#4167"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 70, "Q4_K_M", 2048, 10.0, 35.0, "deepnewz"),
        ("MacBook Pro M3 Max 64GB", "Llama 3.1", 70, "Q4_K_M", 8192, 8.5, 28.0, "deepnewz"),

        # ========== Apple Silicon - M2 Max with context scaling ==========
        ("MacBook Pro M2 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 2048, 66.0, 671.3, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 4096, 62.0, 620.0, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 8192, 56.0, 540.0, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 64GB", "Llama 3.1", 8, "Q4_K_M", 16384, 48.0, 420.0, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 32GB", "Llama 3.1", 8, "Q4_K_M", 2048, 65.0, 650.0, "llama.cpp#4167"),
        ("MacBook Pro M2 Max 32GB", "Llama 3.1", 8, "F16", 2048, 24.5, 748.0, "llama.cpp#4167"),

        # ========== Apple Silicon - M2 Ultra with context scaling ==========
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 2048, 76.3, 1200.0, "gpu-bench-repo"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 4096, 72.0, 1100.0, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 8192, 66.0, 950.0, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 16384, 58.0, 780.0, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 32768, 48.0, 580.0, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 8, "F16", 2048, 41.0, 1401.9, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 70, "Q4_K_M", 2048, 14.0, 82.0, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 70, "Q4_K_M", 8192, 12.0, 65.0, "llama.cpp#4167"),
        ("Mac Studio M2 Ultra 128GB", "Llama 3.1", 70, "Q4_K_M", 16384, 10.5, 52.0, "llama.cpp#4167"),

        # ========== Apple Silicon - M3 Ultra with context scaling ==========
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 8, "Q4_K_M", 2048, 78.0, 1120.0, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 8, "Q4_K_M", 8192, 68.0, 900.0, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 8, "Q4_K_M", 32768, 50.0, 550.0, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 8, "F16", 2048, 42.2, 1121.8, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 70, "Q4_K_M", 2048, 18.0, 95.0, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 70, "Q4_K_M", 8192, 15.0, 75.0, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 70, "Q4_K_M", 16384, 12.5, 58.0, "llama.cpp#4167"),
        ("Mac Studio M3 Ultra 256GB", "Llama 3.1", 70, "Q4_K_M", 32768, 10.0, 42.0, "llama.cpp#4167"),

        # ========== Apple Silicon - M1 Series ==========
        ("MacBook Pro M1 Max 32GB", "Llama 3.1", 8, "Q4_K_M", 2048, 61.2, 530.0, "llama.cpp#4167"),
        ("MacBook Pro M1 Max 32GB", "Llama 3.1", 8, "Q4_K_M", 8192, 52.0, 420.0, "llama.cpp#4167"),
        ("MacBook Pro M1 Max 32GB", "Llama 3.1", 8, "F16", 2048, 23.0, 599.5, "llama.cpp#4167"),
        ("Mac Studio M1 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 2048, 72.0, 900.0, "llama.cpp#4167"),
        ("Mac Studio M1 Ultra 128GB", "Llama 3.1", 8, "Q4_K_M", 8192, 62.0, 720.0, "llama.cpp#4167"),
        ("Mac Studio M1 Ultra 128GB", "Llama 3.1", 8, "F16", 2048, 37.0, 1168.9, "llama.cpp#4167"),

        # ========== Smaller models - 3B class ==========
        ("RTX 4090 24GB", "Llama 3.2", 3, "Q4_K_M", 2048, 220.0, 12000.0, "llama.cpp#10879"),
        ("RTX 4090 24GB", "Llama 3.2", 3, "Q4_K_M", 8192, 180.0, 9500.0, "llama.cpp#10879"),
        ("RTX 4090 24GB", "Llama 3.2", 3, "Q4_K_M", 32768, 120.0, 5500.0, "llama.cpp#10879"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.2", 3, "Q4_K_M", 2048, 150.0, 1200.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Llama 3.2", 3, "Q4_K_M", 8192, 130.0, 950.0, "llama.cpp#4167"),

        # ========== 7B models ==========
        ("RTX 4090 24GB", "Qwen 2.5", 7, "Q4_K_M", 2048, 145.0, 9800.0, "llama.cpp#10879"),
        ("RTX 4090 24GB", "Qwen 2.5", 7, "Q4_K_M", 8192, 125.0, 7500.0, "llama.cpp#10879"),
        ("RTX 4090 24GB", "Qwen 2.5", 7, "Q4_K_M", 32768, 85.0, 4200.0, "llama.cpp#10879"),
        ("MacBook Pro M4 Max 64GB", "Qwen 2.5", 7, "Q4_K_M", 2048, 95.0, 800.0, "llama.cpp#4167"),
        ("MacBook Pro M4 Max 64GB", "Qwen 2.5", 7, "Q4_K_M", 8192, 82.0, 650.0, "llama.cpp#4167"),

        # ========== AMD GPUs ==========
        ("RX 7900 XTX 24GB", "Llama 3.1", 8, "Q4_K_M", 2048, 95.0, 3200.0, "llama.cpp#10879"),
        ("RX 7900 XTX 24GB", "Llama 3.1", 8, "Q4_K_M", 4096, 88.0, 2900.0, "llama.cpp#10879"),
        ("RX 7900 XTX 24GB", "Llama 3.1", 8, "Q4_K_M", 8192, 78.0, 2500.0, "llama.cpp#10879"),
        ("RX 7900 XTX 24GB", "Llama 3.1", 8, "Q4_K_M", 16384, 62.0, 1900.0, "llama.cpp#10879"),
        ("RX 7900 XTX 24GB", "Qwen 2.5", 14, "Q4_K_M", 2048, 62.0, 2200.0, "llama.cpp#10879"),
        ("RX 7900 XTX 24GB", "Qwen 2.5", 14, "Q4_K_M", 8192, 52.0, 1700.0, "llama.cpp#10879"),

        # ========== Professional GPUs ==========
        ("RTX A6000 48GB", "Llama 3.1", 8, "Q4_K_M", 2048, 100.0, 3800.0, "llama.cpp#10879"),
        ("RTX A6000 48GB", "Llama 3.1", 8, "Q4_K_M", 8192, 88.0, 3200.0, "llama.cpp#10879"),
        ("RTX A6000 48GB", "Llama 3.1", 70, "Q4_K_M", 2048, 18.0, 650.0, "llama.cpp#10879"),
        ("RTX A6000 48GB", "Llama 3.1", 70, "Q4_K_M", 8192, 15.0, 500.0, "llama.cpp#10879"),

        # ========== Dual GPU configs ==========
        ("2x RTX 4090 48GB", "Llama 3.1", 70, "Q4_K_M", 2048, 38.0, 1600.0, "llama.cpp#10879"),
        ("2x RTX 4090 48GB", "Llama 3.1", 70, "Q4_K_M", 8192, 32.0, 1300.0, "llama.cpp#10879"),
        ("2x RTX 4090 48GB", "Llama 3.1", 70, "Q4_K_M", 16384, 26.0, 1000.0, "llama.cpp#10879"),

        # ========== NVIDIA A100 80GB PCIe - Context scaling ==========
        ("A100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 2048, 138.0, 5800.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 4096, 130.0, 5200.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 8192, 115.0, 4500.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 16384, 95.0, 3500.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 32768, 72.0, 2500.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 8, "F16", 2048, 54.5, 7504.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 2048, 22.0, 727.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 4096, 20.5, 650.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 8192, 18.0, 550.0, "gpu-bench-repo"),
        ("A100 80GB PCIe", "Qwen 2.5", 32, "Q4_K_M", 2048, 36.0, 1800.0, "databasemart-a100"),
        ("A100 80GB PCIe", "Qwen 2.5", 32, "Q4_K_M", 8192, 32.0, 1400.0, "databasemart-a100"),

        # ========== NVIDIA A100 80GB SXM - Higher bandwidth ==========
        ("A100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 2048, 133.0, 5864.0, "gpu-bench-repo"),
        ("A100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 4096, 125.0, 5200.0, "gpu-bench-repo"),
        ("A100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 8192, 110.0, 4400.0, "gpu-bench-repo"),
        ("A100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 16384, 90.0, 3400.0, "gpu-bench-repo"),
        ("A100 80GB SXM", "Llama 3.1", 70, "Q4_K_M", 2048, 24.0, 797.0, "gpu-bench-repo"),
        ("A100 80GB SXM", "Llama 3.1", 70, "Q4_K_M", 8192, 20.0, 600.0, "gpu-bench-repo"),

        # ========== NVIDIA H100 80GB PCIe - Context scaling ==========
        ("H100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 2048, 144.0, 7760.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 4096, 138.0, 7000.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 8192, 125.0, 6000.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 16384, 105.0, 4800.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 32768, 82.0, 3500.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 8, "Q4_K_M", 65536, 58.0, 2200.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 8, "F16", 2048, 68.0, 10343.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 2048, 25.0, 984.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 4096, 23.5, 880.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 8192, 21.0, 750.0, "gpu-bench-repo"),
        ("H100 80GB PCIe", "Llama 3.1", 70, "Q4_K_M", 16384, 18.0, 580.0, "gpu-bench-repo"),

        # ========== NVIDIA H100 80GB SXM - Highest bandwidth ==========
        ("H100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 2048, 165.0, 9500.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 4096, 158.0, 8500.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 8192, 145.0, 7200.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 16384, 125.0, 5800.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 32768, 98.0, 4200.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 8, "Q4_K_M", 65536, 70.0, 2800.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 8, "F16", 2048, 82.0, 12500.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 70, "Q4_K_M", 2048, 32.0, 1200.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 70, "Q4_K_M", 4096, 30.0, 1050.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 70, "Q4_K_M", 8192, 27.0, 880.0, "valdi-h100"),
        ("H100 80GB SXM", "Llama 3.1", 70, "Q4_K_M", 16384, 23.0, 680.0, "valdi-h100"),
        ("H100 80GB SXM", "Qwen 2.5", 32, "Q4_K_M", 2048, 52.0, 3200.0, "valdi-h100"),
        ("H100 80GB SXM", "Qwen 2.5", 32, "Q4_K_M", 8192, 45.0, 2500.0, "valdi-h100"),
        ("H100 80GB SXM", "Qwen 2.5", 72, "Q4_K_M", 2048, 22.0, 850.0, "valdi-h100"),

        # ========== NVIDIA H200 141GB SXM - Massive bandwidth ==========
        ("H200 141GB SXM", "Llama 3.1", 8, "Q4_K_M", 2048, 195.0, 12000.0, "nvidia-h100-blog"),
        ("H200 141GB SXM", "Llama 3.1", 8, "Q4_K_M", 8192, 175.0, 9500.0, "nvidia-h100-blog"),
        ("H200 141GB SXM", "Llama 3.1", 8, "Q4_K_M", 32768, 130.0, 6000.0, "nvidia-h100-blog"),
        ("H200 141GB SXM", "Llama 3.1", 70, "Q4_K_M", 2048, 42.0, 1600.0, "nvidia-h100-blog"),
        ("H200 141GB SXM", "Llama 3.1", 70, "Q4_K_M", 8192, 38.0, 1300.0, "nvidia-h100-blog"),
        ("H200 141GB SXM", "Llama 3.1", 70, "Q4_K_M", 32768, 28.0, 850.0, "nvidia-h100-blog"),
        ("H200 141GB SXM", "Llama 3.1", 70, "F16", 2048, 18.0, 2000.0, "nvidia-h100-blog"),

        # ========== RTX 5090 32GB - Context scaling ==========
        ("RTX 5090 32GB", "Llama 3.1", 8, "Q4_K_M", 4096, 186.0, 10400.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Llama 3.1", 8, "Q4_K_M", 8192, 170.0, 8750.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Llama 3.1", 8, "Q4_K_M", 16384, 140.0, 6200.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Llama 3.1", 8, "Q4_K_M", 32768, 112.0, 4100.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Llama 3.1", 8, "Q4_K_M", 65536, 78.0, 2500.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Llama 3.1", 8, "Q4_K_M", 131072, 48.0, 1400.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 124.0, 6500.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Qwen 2.5", 14, "Q4_K_M", 8192, 115.0, 5600.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Qwen 2.5", 14, "Q4_K_M", 32768, 82.0, 2900.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Qwen 2.5", 32, "Q4_K_M", 4096, 61.0, 2930.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Qwen 2.5", 32, "Q4_K_M", 8192, 56.0, 2530.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Qwen 2.5", 32, "Q4_K_M", 32768, 44.0, 1450.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Mixtral", 46.7, "Q4_K_M", 4096, 234.0, 6630.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Mixtral", 46.7, "Q4_K_M", 8192, 170.0, 5800.0, "hardware-corner-5090"),
        ("RTX 5090 32GB", "Mixtral", 46.7, "Q4_K_M", 32768, 111.0, 2880.0, "hardware-corner-5090"),

        # ========== 2x RTX 5090 64GB ==========
        ("2x RTX 5090 64GB", "Llama 3.1", 70, "Q4_K_M", 2048, 27.0, 2200.0, "databasemart-5090"),
        ("2x RTX 5090 64GB", "Llama 3.1", 70, "Q4_K_M", 8192, 24.0, 1800.0, "databasemart-5090"),
        ("2x RTX 5090 64GB", "Llama 3.1", 70, "Q4_K_M", 16384, 20.0, 1400.0, "databasemart-5090"),

        # ========== DGX Spark 128GB (GB10) - Unified memory ==========
        ("DGX Spark 128GB", "Llama 3.1", 8, "Q4_K_M", 2048, 59.0, 3600.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Llama 3.1", 8, "Q4_K_M", 4096, 55.0, 3200.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Llama 3.1", 8, "Q4_K_M", 8192, 48.0, 2700.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Llama 3.1", 8, "Q4_K_M", 32768, 32.0, 1500.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Llama 3.1", 70, "Q4_K_M", 2048, 18.0, 817.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Llama 3.1", 70, "Q4_K_M", 4096, 16.0, 720.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Llama 3.1", 70, "Q4_K_M", 8192, 14.0, 600.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Qwen 2.5", 32, "Q4_K_M", 2048, 28.0, 1200.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Qwen 2.5", 32, "Q4_K_M", 8192, 24.0, 950.0, "lmsys-dgx-spark"),
        ("DGX Spark 128GB", "Qwen 2.5", 72, "Q4_K_M", 2048, 12.0, 480.0, "lmsys-dgx-spark"),

        # ========== 2x DGX Spark 256GB ==========
        ("2x DGX Spark 256GB", "Llama 3.1", 70, "Q4_K_M", 2048, 23.0, 950.0, "lmsys-dgx-spark"),
        ("2x DGX Spark 256GB", "Llama 3.1", 70, "Q4_K_M", 8192, 20.0, 780.0, "lmsys-dgx-spark"),
        ("2x DGX Spark 256GB", "Llama 3.1", 405, "Q4_K_M", 2048, 5.5, 180.0, "lmsys-dgx-spark"),

        # ========== LocalScore.ai - Real TTFT measurements (llamafile backend) ==========
        # NOTE: URLs 1874-1879 were incorrectly mapped - removed invalid entries
        # Valid entries below from localscore-github sqlite dump

        # ========== LocalScore GitHub db.sqlite - Real TTFT measurements (llamafile backend) ==========
        # RTX 3090 benchmarks
        ("RTX 3090 24GB", "Llama 3.2", 1, "Q4_K_M", 4096, 330.0, 14061.1, "localscore-github", 95),
        ("RTX 3090 24GB", "Llama 3.1", 8, "Q4_K_M", 4096, 103.1, 3552.3, "localscore-github", 358),
        ("RTX 3090 24GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 59.9, 2005.3, "localscore-github", 640),
        # RTX 4060 Ti benchmarks
        ("RTX 4060 Ti 16GB", "Llama 3.2", 1, "Q4_K_M", 4096, 218.0, 9337.1, "localscore-github", 161),
        ("RTX 4060 Ti 16GB", "Llama 3.1", 8, "Q4_K_M", 4096, 49.3, 2246.6, "localscore-github", 612),
        ("RTX 4060 Ti 16GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 26.9, 1266.2, "localscore-github", 1074),
        # Apple Silicon benchmarks
        ("MacBook Pro M4 Max 128GB", "Llama 3.2", 1, "Q4_K_M", 4096, 184.2, 3893.3, "localscore-github", 301),
        ("MacBook Pro M4 Max 128GB", "Llama 3.1", 8, "Q4_K_M", 4096, 51.6, 607.3, "localscore-github", 1994),
        ("MacBook Pro M1 Pro 32GB", "Llama 3.2", 1, "Q4_K_M", 4096, 80.4, 1203.6, "localscore-github", 1002),

        # ========== LocalScore.ai Result Pages - Real TTFT measurements ==========
        # RTX 5090 32GB - https://www.localscore.ai/result/175
        ("RTX 5090 32GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 65.1, 4787.0, "https://www.localscore.ai/result/175", 279),
        # RTX 4070 Ti Super 16GB - https://www.localscore.ai/result/150
        ("RTX 4070 Ti Super 16GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 53.6, 2526.0, "https://www.localscore.ai/result/150", 521),
        # RTX 3070 Ti 8GB - https://www.localscore.ai/result/125
        ("RTX 3070 Ti 8GB", "Llama 3.1", 8, "Q4_K_M", 4096, 83.2, 2509.0, "https://www.localscore.ai/result/125", 520),
        # RTX 3060 Ti 8GB - https://www.localscore.ai/result/225
        ("RTX 3060 Ti 8GB", "Llama 3.2", 1, "Q4_K_M", 4096, 182.0, 8050.0, "https://www.localscore.ai/result/225", 169),
        # RTX 3060 12GB - https://www.localscore.ai/result/800
        ("RTX 3060 12GB", "Llama 3.1", 8, "Q4_K_M", 4096, 49.7, 1400.0, "https://www.localscore.ai/result/800", 933),
        # RTX 3060 12GB - https://www.localscore.ai/result/500
        ("RTX 3060 12GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 26.0, 802.7, "https://www.localscore.ai/result/500", 1614),
        # RTX A6000 48GB - https://www.localscore.ai/result/50
        ("RTX A6000 48GB", "Llama 3.2", 1, "Q4_K_M", 4096, 315.0, 13191.0, "https://www.localscore.ai/result/50", 102),
        # MacBook Pro M3 Max 128GB - https://www.localscore.ai/result/750
        ("MacBook Pro M3 Max 128GB", "Llama 3.1", 8, "Q4_K_M", 4096, 45.6, 591.0, "https://www.localscore.ai/result/750", 2030),
        # MacBook Pro M1 Max 32GB - https://www.localscore.ai/result/900
        ("MacBook Pro M1 Max 32GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 18.1, 178.0, "https://www.localscore.ai/result/900", 7130),
        # MacBook Pro M1 16GB - https://www.localscore.ai/result/950
        ("MacBook Pro M1 16GB", "Llama 3.2", 1, "Q4_K_M", 4096, 38.6, 536.0, "https://www.localscore.ai/result/950", 2429),
        # RX 7900 XTX 24GB - https://www.localscore.ai/result/1000
        ("RX 7900 XTX 24GB", "Qwen 2.5", 14, "Q4_K_M", 4096, 27.95, 501.86, "https://www.localscore.ai/result/1000", 2570),

        # ========== YouTube Channel Benchmarks (Alex Ziskind) ==========
        # Source: "Zero to Hero LLMs with M3 Max BEAST" - https://www.youtube.com/watch?v=0RRsjHprna4
        # OpenHermes 7B (Mistral-based) F16 on M3 Max 64GB - 237 tok/s generation
        ("MacBook Pro M3 Max 64GB", "Mistral", 7, "F16", 2048, 237.0, 800.0, "https://www.youtube.com/watch?v=0RRsjHprna4"),

        # Source: "DeepSeek on Apple Silicon in depth | 4 MacBooks Tested" - https://www.youtube.com/watch?v=jdgy9YUSv0s
        # DeepSeek R1 Distill 1.5B Q4_K_M benchmarks
        ("MacBook Air M3 8GB", "DeepSeek", 1.5, "Q4_K_M", 2048, 54.0, 400.0, "https://www.youtube.com/watch?v=jdgy9YUSv0s"),
        ("MacBook Air M2 8GB", "DeepSeek", 1.5, "Q4_K_M", 2048, 47.0, 350.0, "https://www.youtube.com/watch?v=jdgy9YUSv0s"),
        ("MacBook Pro M4 Max 128GB", "DeepSeek", 1.5, "Q4_K_M", 2048, 182.0, 1500.0, "https://www.youtube.com/watch?v=jdgy9YUSv0s"),
        # DeepSeek R1 Distill Llama 8B benchmarks
        ("MacBook Pro M4 Max 128GB", "DeepSeek", 8, "Q4_K_M", 2048, 68.76, 600.0, "https://www.youtube.com/watch?v=jdgy9YUSv0s"),
        # DeepSeek R1 Distill 14B benchmark (M1 16GB from video)
        ("MacBook Pro M1 16GB", "DeepSeek", 14, "Q4_K_M", 2048, 6.2, 50.0, "https://www.youtube.com/watch?v=jdgy9YUSv0s"),
        # DeepSeek R1 Distill Llama 70B benchmark
        ("MacBook Pro M4 Max 128GB", "DeepSeek", 70, "Q4_K_M", 2048, 9.7, 80.0, "https://www.youtube.com/watch?v=jdgy9YUSv0s"),

        # Source: "LLMs on RTX 4090 Laptop vs Desktop" - https://www.youtube.com/watch?v=2hi1VoOI00g
        # Llama 3.2 1B benchmarks
        ("RTX 4090 24GB", "Llama 3.2", 1, "Q4_K_M", 2048, 313.0, 15000.0, "https://www.youtube.com/watch?v=2hi1VoOI00g"),
        ("RTX 4090 Laptop 16GB", "Llama 3.2", 1, "Q4_K_M", 2048, 225.0, 11000.0, "https://www.youtube.com/watch?v=2hi1VoOI00g"),
        # Llama 3.2 3B benchmarks
        ("RTX 4090 24GB", "Llama 3.2", 3, "Q4_K_M", 2048, 201.0, 10000.0, "https://www.youtube.com/watch?v=2hi1VoOI00g"),
        ("RTX 4090 Laptop 16GB", "Llama 3.2", 3, "Q4_K_M", 2048, 151.0, 7500.0, "https://www.youtube.com/watch?v=2hi1VoOI00g"),
        # Qwen 2.5 Coder 32B benchmark
        ("RTX 4090 24GB", "Qwen 2.5", 32, "Q4_K_M", 2048, 27.0, 1500.0, "https://www.youtube.com/watch?v=2hi1VoOI00g"),

        # Source: "M3 Ultra vs RTX 5090 | The Final Battle" - https://www.youtube.com/watch?v=nwIZ5VI3Eus
        # Gemma 3 4B benchmarks
        ("RTX 5090 32GB", "Gemma 3", 4, "Q4_K_M", 2048, 97.0, 5000.0, "https://www.youtube.com/watch?v=nwIZ5VI3Eus"),
        ("Mac Studio M3 Ultra 512GB", "Gemma 3", 4, "Q4_K_M", 2048, 100.0, 800.0, "https://www.youtube.com/watch?v=nwIZ5VI3Eus"),
        # Gemma 3 27B benchmarks
        ("RTX 5090 32GB", "Gemma 3", 27, "Q4_K_M", 2048, 41.0, 2500.0, "https://www.youtube.com/watch?v=nwIZ5VI3Eus"),
        ("Mac Studio M3 Ultra 512GB", "Gemma 3", 27, "Q4_K_M", 2048, 28.0, 200.0, "https://www.youtube.com/watch?v=nwIZ5VI3Eus"),
        # DeepSeek R1 70B benchmark (full GPU on M3 Ultra)
        ("Mac Studio M3 Ultra 512GB", "DeepSeek", 70, "Q4_K_M", 2048, 13.0, 100.0, "https://www.youtube.com/watch?v=nwIZ5VI3Eus"),

        # Source: "Not even close | LLMs on RTX5090 vs others" - https://www.youtube.com/watch?v=AScA7qJUIDc
        # Qwen 2.5 Coder 32B benchmarks
        ("RTX 5090 32GB", "Qwen 2.5", 32, "Q4_K_M", 2048, 62.0, 3500.0, "https://www.youtube.com/watch?v=AScA7qJUIDc"),
        ("RTX 5090 Laptop 24GB", "Qwen 2.5", 32, "Q4_K_M", 2048, 31.0, 1800.0, "https://www.youtube.com/watch?v=AScA7qJUIDc"),
        # DeepSeek R1 Distill Llama 8B benchmarks
        ("RTX 5090 32GB", "DeepSeek", 8, "Q4_K_M", 2048, 202.0, 12000.0, "https://www.youtube.com/watch?v=AScA7qJUIDc"),
        ("RTX 5090 Laptop 24GB", "DeepSeek", 8, "Q4_K_M", 2048, 104.0, 6000.0, "https://www.youtube.com/watch?v=AScA7qJUIDc"),
        ("RTX 5080 16GB", "DeepSeek", 8, "Q4_K_M", 2048, 132.0, 8000.0, "https://www.youtube.com/watch?v=AScA7qJUIDc"),
        ("RTX 5060 Ti 16GB", "DeepSeek", 8, "Q4_K_M", 2048, 73.8, 4500.0, "https://www.youtube.com/watch?v=AScA7qJUIDc"),

        # Source: Koyeb GPU Benchmarks - https://www.koyeb.com/docs/hardware/gpu-benchmarks
        # L40S benchmarks (single-user inference, lower range values)
        ("L40S 48GB", "Llama 3.1", 8, "F16", 2048, 43.79, 1124.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
        ("L40S 48GB", "DeepSeek", 8, "F16", 2048, 40.60, 1116.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
        ("L40S 48GB", "Qwen 2.5", 7, "F16", 2048, 45.81, 1050.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
        # H100 SXM benchmarks from Koyeb (complementing existing data)
        ("H100 80GB SXM", "DeepSeek", 8, "F16", 2048, 87.89, 2442.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
        ("H100 80GB SXM", "Qwen 2.5", 7, "F16", 2048, 87.59, 1959.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
        # A100 SXM benchmarks from Koyeb
        ("A100 80GB SXM", "DeepSeek", 8, "F16", 2048, 75.38, 1900.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
        ("A100 80GB SXM", "Qwen 2.5", 7, "F16", 2048, 81.64, 1942.0, "https://www.koyeb.com/docs/hardware/gpu-benchmarks"),
    ]

    with get_cursor() as cursor:
        for benchmark in benchmarks_data:
            # Handle both 8-element (no ttft) and 9-element (with ttft) tuples
            hw_name = benchmark[0]
            model_family = benchmark[1]
            model_size = benchmark[2]
            quant_type = benchmark[3]
            context_len = benchmark[4]
            gen_tps = benchmark[5]
            prompt_tps = benchmark[6]
            source = benchmark[7]
            ttft_ms = benchmark[8] if len(benchmark) > 8 else None

            cursor.execute("SELECT id FROM hardware WHERE name = ?", (hw_name,))
            hw_row = cursor.fetchone()
            if not hw_row:
                print(f"Warning: Hardware '{hw_name}' not found for benchmark. Skipping.")
                continue
            hardware_id = hw_row[0]

            cursor.execute(
                "SELECT id FROM models WHERE family = ? AND size_b = ?",
                (model_family, model_size),
            )
            model_row = cursor.fetchone()
            if not model_row:
                print(f"Warning: Model '{model_family} {model_size}B' not found for benchmark. Skipping.")
                continue
            model_id = model_row[0]

            cursor.execute(
                "SELECT id FROM quantizations WHERE model_id = ? AND quant_type = ?",
                (model_id, quant_type),
            )
            quant_row = cursor.fetchone()
            if not quant_row:
                print(f"Warning: Quantization '{quant_type}' for model '{model_family} {model_size}B' not found. Skipping.")
                continue
            quantization_id = quant_row[0]

            # Use source URL from SOURCES dict, or use raw URL if it's already a URL
            source_url = SOURCES.get(source, source)

            cursor.execute(
                """
                INSERT OR IGNORE INTO benchmarks
                (hardware_id, quantization_id, prompt_tokens, generation_tokens,
                 context_length, generation_tps, prompt_tps, ttft_ms, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (hardware_id, quantization_id, 512, 128, context_len, gen_tps, prompt_tps, ttft_ms, source_url),
            )


def seed_ocr_tools():
    """Seed OCR tools data."""
    # (name, vendor, tool_type, requires_gpu, min_vram_gb, base_ram_gb, description, source)
    ocr_tools = [
        ("Docling v2", "IBM", "neural", 0, None, 4.0,
         "Fast document conversion with layout analysis and table extraction",
         "https://github.com/DS4SD/docling"),
        ("Tesseract 5.0", "Google", "traditional", 0, None, 0.3,
         "Classic OCR engine, CPU-only, supports 100+ languages",
         "https://github.com/tesseract-ocr/tesseract"),
        ("EasyOCR", "JaidedAI", "neural", 1, 4.0, 2.0,
         "Ready-to-use OCR with 80+ languages, GPU accelerated",
         "https://github.com/JaidedAI/EasyOCR"),
        ("olmOCR", "Allenai", "neural", 1, 20.0, 8.0,
         "Vision transformer OCR, high accuracy but requires significant VRAM",
         "https://github.com/allenai/olmocr"),
    ]

    with get_cursor() as cursor:
        for tool in ocr_tools:
            cursor.execute(
                """
                INSERT OR IGNORE INTO ocr_tools
                (name, vendor, tool_type, requires_gpu, min_vram_gb, base_ram_gb, description, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tool,
            )


def seed_ocr_benchmarks():
    """Seed OCR benchmarks data."""
    # (hardware_name, tool_name, pages_per_second, ram_usage_gb, vram_usage_gb,
    #  cpu_cores_used, is_gpu_accelerated, document_type, source)
    ocr_benchmarks = [
        # Docling v2 benchmarks
        ("RTX 4090 24GB", "Docling v2", 20.0, 4.0, 8.0, 4, 1, "mixed",
         "https://github.com/DS4SD/docling"),
        ("MacBook Pro M4 Max 64GB", "Docling v2", 15.0, 4.0, None, 8, 0, "mixed",
         "https://github.com/DS4SD/docling"),
        ("RTX 3090 24GB", "Docling v2", 16.0, 4.0, 8.0, 4, 1, "mixed",
         "https://github.com/DS4SD/docling"),
        ("RTX 4060 Ti 16GB", "Docling v2", 10.0, 4.0, 6.0, 4, 1, "mixed",
         "https://github.com/DS4SD/docling"),
        ("MacBook Pro M1 Max 32GB", "Docling v2", 8.0, 4.0, None, 8, 0, "mixed",
         "https://github.com/DS4SD/docling"),

        # Tesseract benchmarks (CPU only)
        ("MacBook Pro M4 Max 64GB", "Tesseract 5.0", 0.15, 0.3, None, 1, 0, "pdf_scanned",
         "https://github.com/tesseract-ocr/tesseract"),
        ("MacBook Pro M1 Max 32GB", "Tesseract 5.0", 0.12, 0.3, None, 1, 0, "pdf_scanned",
         "https://github.com/tesseract-ocr/tesseract"),
        ("RTX 4090 24GB", "Tesseract 5.0", 0.13, 0.3, None, 1, 0, "pdf_scanned",
         "https://github.com/tesseract-ocr/tesseract"),

        # EasyOCR benchmarks
        ("RTX 4090 24GB", "EasyOCR", 2.0, 2.0, 4.0, 2, 1, "mixed",
         "https://github.com/JaidedAI/EasyOCR"),
        ("RTX 3090 24GB", "EasyOCR", 1.5, 2.0, 4.0, 2, 1, "mixed",
         "https://github.com/JaidedAI/EasyOCR"),
        ("MacBook Pro M4 Max 64GB", "EasyOCR", 0.8, 2.0, None, 4, 0, "mixed",
         "https://github.com/JaidedAI/EasyOCR"),

        # olmOCR benchmarks (high-end GPU only)
        ("RTX 4090 24GB", "olmOCR", 5.0, 8.0, 20.0, 4, 1, "pdf_scanned",
         "https://github.com/allenai/olmocr"),
        ("H100 80GB SXM", "olmOCR", 12.0, 8.0, 20.0, 4, 1, "pdf_scanned",
         "https://github.com/allenai/olmocr"),
    ]

    with get_cursor() as cursor:
        for benchmark in ocr_benchmarks:
            hw_name = benchmark[0]
            tool_name = benchmark[1]

            cursor.execute("SELECT id FROM hardware WHERE name = ?", (hw_name,))
            hw_row = cursor.fetchone()
            if not hw_row:
                continue
            hardware_id = hw_row[0]

            cursor.execute("SELECT id FROM ocr_tools WHERE name = ?", (tool_name,))
            tool_row = cursor.fetchone()
            if not tool_row:
                continue
            tool_id = tool_row[0]

            cursor.execute(
                """
                INSERT OR IGNORE INTO ocr_benchmarks
                (hardware_id, tool_id, pages_per_second, ram_usage_gb, vram_usage_gb,
                 cpu_cores_used, is_gpu_accelerated, document_type, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (hardware_id, tool_id, benchmark[2], benchmark[3], benchmark[4],
                 benchmark[5], benchmark[6], benchmark[7], benchmark[8]),
            )


def seed_database():
    """Initialize and seed the entire database."""
    init_database()
    seed_hardware()
    seed_models()
    seed_quantizations()
    seed_benchmarks()
    seed_ocr_tools()
    seed_ocr_benchmarks()


if __name__ == "__main__":
    seed_database()
    print("Database seeded successfully!")

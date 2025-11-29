# Changelog

All notable changes to this project will be documented in this file.

## [0.3.5] - 2025-11-29

### Added
- **Document Processing mode** - new third mode for OCR and document parsing simulation:
  - Estimate processing time for PDFs using different OCR tools
  - Shows pages per second, total time, RAM/VRAM requirements
  - Supports three document types: text PDF, scanned PDF, and mixed
- OCR tools database with 4 popular tools:
  - Docling v2 (IBM, neural, GPU-accelerated)
  - Tesseract 5.0 (Google, traditional, CPU-only)
  - EasyOCR (JaidedAI, neural, GPU-accelerated)
  - olmOCR (Open LM, neural, GPU-accelerated)
- New API endpoints:
  - `GET /api/v1/ocr-tools` - list available OCR tools
  - `POST /api/v1/simulate-ocr` - simulate OCR performance
- Database tables for OCR: `ocr_tools`, `ocr_benchmarks`
- Frontend OCR panel with hardware/tool selection, page count, and document type

### Changed
- Mode switch UI now supports 3 buttons instead of 2
- `nearest_benchmarks` now uses **hardware similarity** instead of TPS/TTFT distance:
  - Matches hardware by bandwidth, memory, and model size proximity
  - Shows benchmarks from similar hardware configurations
  - Fixes issue where M4 Max showed M4 Pro (4x different bandwidth) as "nearest"

### Fixed
- Confidence chart now shows relevant benchmarks from similar hardware
- Nearest benchmark selection considers hardware specs instead of just performance metrics

## [0.3.4] - 2025-11-29

### Added
- Clickable confidence score link opens explanation modal
- Chart visualization showing estimated point vs 4 nearest real benchmarks (X: TPS, Y: TTFT)
- Confidence calculation explanation in modal
- `nearest_benchmarks` field in simulation API response
- `get_nearest_benchmarks()` database query

### Changed
- Confidence display now a clickable link instead of plain text

## [0.3.3] - 2025-11-29

### Added
- Real Time to First Token (TTFT) measurements from LocalScore.ai website (6 benchmarks):
  - Mac Mini M4 Pro 24GB: Llama 3.1 8B @ 4K ctx - TTFT 4147ms
  - RTX 4070 Super 12GB: Llama 3.2 1B @ 4K ctx - TTFT 109ms
  - RTX 4090 24GB: Llama 3.2 1B @ 4K ctx - TTFT 52ms
  - RTX 4090 24GB: Llama 3.1 8B @ 4K ctx - TTFT 170ms
  - RTX 4090 24GB: Qwen 2.5 14B @ 4K ctx - TTFT 304ms
  - DGX Spark 128GB: Qwen 2.5 14B @ 4K ctx - TTFT 1243ms
- Real TTFT measurements from LocalScore GitHub db.sqlite (9 benchmarks):
  - RTX 3090: Llama 3.2 1B (95ms), Llama 3.1 8B (358ms), Qwen 2.5 14B (640ms)
  - RTX 4060 Ti: Llama 3.2 1B (161ms), Llama 3.1 8B (612ms), Qwen 2.5 14B (1074ms)
  - M4 Max 128GB: Llama 3.2 1B (301ms), Llama 3.1 8B (1994ms)
  - M1 Pro 32GB: Llama 3.2 1B (1002ms)
- Real TTFT measurements from LocalScore.ai result pages (11 benchmarks):
  - RTX 5090: Qwen 2.5 14B (279ms)
  - RTX 4070 Ti Super: Qwen 2.5 14B (521ms)
  - RTX 3070 Ti: Llama 3.1 8B (520ms)
  - RTX 3060 Ti: Llama 3.2 1B (169ms)
  - RTX 3060: Llama 3.1 8B (933ms), Qwen 2.5 14B (1614ms)
  - RTX A6000: Llama 3.2 1B (102ms)
  - M3 Max 64GB: Llama 3.1 8B (2030ms)
  - M1 Max 32GB: Qwen 2.5 14B (7130ms)
  - M1 16GB: Llama 3.2 1B (2429ms)
  - RX 7900 XTX: Qwen 2.5 14B (2570ms)
- Individual benchmark result URLs from LocalScore.ai stored in source field

### Changed
- Benchmarks: 181 → 207 data points (26 new with TTFT)
- Benchmark data format now supports optional ttft_ms field

## [0.3.2] - 2025-11-29

### Fixed
- TTFT calculation now uses correct prompt_tps ratios based on GPU cores:
  - Apple Silicon (unified, <3000 cores): 10x generation speed
  - DGX Spark (unified, 5000+ cores): 50x generation speed
  - RTX 4090/5090 (10000+ cores): 70x generation speed
  - H100/A100 datacenter (14000+ cores): 120x generation speed
- Benchmark matching now correctly selects by context_length instead of prompt_tokens
- Memory indicator label updated to "Model and prompt fit in memory"

### Changed
- Prompt processing speed estimation uses hardware-aware ratios derived from 173 benchmarks
- Previous fixed ratios (4x unified, 40x dedicated) replaced with core-count-based formula

## [0.3.1] - 2025-11-29

### Added
- Comprehensive NVIDIA datacenter GPU support:
  - RTX 50 Series: RTX 5090, 5080, 5070 Ti, 5070, 2x RTX 5090
  - A100 variants: 40GB PCIe, 80GB PCIe, 80GB SXM, 2x/4x/8x multi-GPU configs
  - H100 variants: 80GB PCIe, 80GB SXM, 2x/4x/8x multi-GPU configs (DGX H100)
  - H200 141GB SXM
  - DGX Spark (GB10 Grace Blackwell): 128GB and 2x 256GB configs
- 80 new benchmark data points with context lengths: 2K, 4K, 8K, 16K, 32K, 65K, 131K tokens
- New benchmark sources:
  - hardware-corner.net (RTX 5090)
  - VALDI docs (H100)
  - ORI blog (A100/H100)
  - DatabaseMart (A100, RTX 5090)
  - RunPod blog (RTX 5090)
  - LMSYS (DGX Spark)
  - NVIDIA developer blog (H100 TensorRT-LLM)
- Custom hardware form now pre-fills specs from selected preset hardware

### Changed
- Hardware: 90 → 112 configurations (22 new NVIDIA entries)
- Benchmarks: 101 → 181 data points
- ML predictor R² improved: 0.753 → 0.874

## [0.3.0] - 2025-11-29

### Added
- Context-aware performance predictions using ML model trained on 101 benchmark data points
- Benchmark data with context lengths from 2K to 131K tokens
- Source URLs stored in database for all benchmark data:
  - hardware-corner.net RTX 4090 benchmarks
  - llama.cpp GitHub discussions (#4167, #10879)
  - GPU benchmark repository data
- Context length selector: Short (2K), Medium (8K), Large (32K), Max (1M)
- Custom model specification form in "I have hardware" mode
- Dynamic KV cache memory calculation based on context length
- Default form values for immediate "Simulate" clicks
- Tab switching syncs model selection between modes

### Changed
- ML predictor now uses context_length as feature (13.5% importance)
- KV cache formula: `kv_cache_gb = (params_b / 7) * (context_length / 4096) * 0.5`
- Generation speed correctly decreases with larger context windows
- Benchmarks: 61 → 101 data points with context variation

## [0.2.1] - 2025-11-29

### Added
- Custom hardware specification form in "I have hardware" mode
  - Toggle checkbox to switch between preset and custom hardware
  - Input fields for: name (optional), RAM/VRAM (GB), bandwidth (GB/s), GPU cores (optional)
  - Memory type selection: Unified (Apple Silicon) or Dedicated VRAM (discrete GPU)
- Frontend form validation and disabled states for custom mode

## [0.2.0] - 2025-11-29

### Added
- ML-based performance predictor using GradientBoostingRegressor (R²=0.753)
- Confidence scoring based on distance from training data points
- Training script: backend/src/predictor/train.py

### Data Expansion
- Hardware: 32 → 90 configurations
  - Full Apple Silicon lineup: M1, M1 Pro/Max/Ultra, M2, M2 Pro/Max/Ultra, M3, M3 Pro/Max/Ultra, M4, M4 Pro/Max
  - Mac Studio M3 Ultra 512GB, 256GB, 128GB, 64GB variants
  - MacBook Pro M4 Max 36GB/48GB/64GB/128GB
  - Mac Studio M4 Max 36GB/64GB/128GB
  - Updated NVIDIA lineup: RTX 4090, 4080, 3090, A100, H100
  - AMD: RX 7900 XTX, MI300X
- Models: 22 → 29 LLMs
  - Added: Llama 3.3 70B, DeepSeek V3 671B, Command R/R+ 35B/104B, Phi-4, Yi-1.5
- Quantizations: 8 → 11 types per model
  - Added IQ variants: IQ2_XXS, IQ3_S, IQ4_XS
- Benchmarks: 27 → 61 real measurements from llama.cpp discussions

### Changed
- Predictor now uses ML model with fallback to theoretical formula
- Confidence scores now range 0.4-0.92 based on proximity to training data

## [0.1.0] - 2025-11-29

### Added
- Complete MVP with two modes: "I have hardware" and "I want a model"
- FastAPI backend with SQLite in-memory database
- REST API endpoints: /hardware, /models, /simulate, /search
- Theoretical performance predictor based on memory bandwidth
- Seed data: 32 hardware configs (Apple Silicon, NVIDIA, AMD)
- Seed data: 22 LLM models with 8 quantization levels each
- Seed data: 27 real benchmark measurements
- React frontend with Vite build system
- Token streaming animation for visualization
- Hardware search with minimum requirements calculation

### Backend
- FastAPI with async lifespan for database initialization
- Pydantic schemas for request/response validation
- SQLite database with proper indexes
- Memory-bound performance formula: tps = (bandwidth * efficiency) / model_size
- Confidence scoring based on prediction distance from known benchmarks

### Frontend
- Mode switch between "I have hardware" and "I want a model"
- Hardware/Model selection with cascading dropdowns
- Simulation results with memory bar and token animation
- Hardware search results grouped by vendor
- Responsive design with Inter font

## [Unreleased]

### Added
- Initial project structure
- Configuration files for Python 3.12 and Node 20
- README, LICENSE, .gitignore

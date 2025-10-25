# Query Reformulation Research Repository

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive repository for query reformulation research, implementing **several established baselines** and introducing **ReFormeR**, a novel pattern-based approach for improving information retrieval through intelligent query reformulation.

## 📁 Repository Structure

```
query-reformulation/
├── baselines/                          # Baseline query reformulation methods
│   ├── __init__.py
│   ├── flanqr/                         # FlanQR baseline
│   │   ├── __init__.py
│   │   └── flanqr.py
│   ├── genqr_ensemble/                 # GenQR-Ensemble baseline
│   │   ├── __init__.py
│   │   └── genqr_ensemble.py
│   ├── mugi/                           # MuGI baseline
│   │   ├── __init__.py
│   │   └── mugi.py
│   ├── qa_expand/                      # QA-EXPAND baseline
│   │   ├── __init__.py
│   │   └── qa_expand.py
│   └── query2doc/                      # Query2Doc variants
│       ├── __init__.py
│       ├── query2doc_CoT.py            # Chain-of-Thought variant
│       ├── query2doc_FS.py             # Few-Shot variant
│       └── query2doc_ZS.py             # Zero-Shot variant
├── data/                               # Pre-processed datasets
│   ├── diamond_dataset.tsv             # Diamond reformulation pairs for pattern extraction
│   ├── dlhard/                         # DL-Hard dataset
│   │   ├── reformulated_queries.tsv
│   │   └── run_reformulated_queries.tsv
│   ├── trecdl2019/                     # TREC DL 2019 dataset
│   │   ├── reformulated_queries.tsv
│   │   └── run_reformulated_queries.tsv
│   └── trecdl2020/                     # TREC DL 2020 dataset
│       ├── reformulated_queries.tsv
│       └── run_reformulated_queries.tsv
├── reformer/                           # ReFormeR pattern-based approach
│   ├── __init__.py
│   ├── core/                           # Core ReFormeR implementation
│   │   ├── __init__.py
│   │   └── reformer.py                 # Main ReFormeR class
│   ├── patterns/                       # Pattern extraction and management
│   │   ├── __init__.py
│   │   ├── extracted_patterns.json     # Pre-extracted patterns
│   │   ├── iterative_pattern_extraction.py
│   │   └── query_reformulation_prompts.py
│   ├── prompt_manager.py               # Centralized prompt management
│   └── prompts.json                    # All LLM prompts in JSON format
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🎯 Overview

This repository provides a complete framework for query reformulation research, featuring:

- **ReFormeR**: Novel query reformulation approach (our main contribution)
- **Several Baseline Methods**: MuGI, QA-EXPAND, GenQR-Ensemble, FlanQR, and Query2Doc variants
- **Pattern Extraction**: Tools for learning reformulation patterns from data
- **vLLM Support**: GPU-optimized inference with Qwen/Qwen2.5-7B-Instruct
- **Centralized Prompts**: All LLM prompts managed in `prompts.json`

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/queryreformulation/ReFormeR.git
cd ReFormeR

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
```

## 🎯 ReFormeR

ReFormeR is our novel contribution that learns and applies reformulation patterns systematically:

1. **Learns Patterns**: Extracts reformulation patterns from query pairs
2. **Document-Based Selection**: Uses retrieved documents to select appropriate patterns
3. **Systematic Application**: Applies patterns systematically for consistent reformulation

### Core Components

- **`reformer/core/reformer.py`**: Main ReFormeR implementation
- **`reformer/patterns/iterative_pattern_extraction.py`**: Pattern learning from data
- **`reformer/patterns/query_reformulation_prompts.py`**: Prompt engineering and data structures
- **`reformer/prompt_manager.py`**: Centralized prompt management
- **`reformer/prompts.json`**: All LLM prompts in JSON format

### ReFormeR Usage

#### Python API

```python
from reformer import Reformulator

# Initialize ReFormeR
reformer = Reformulator(
    bm25_run_path="data/trecdl2019/run_original_queries.tsv",
    collection_path="data/trecdl2019/collection.tsv",
    patterns_path="reformer/patterns/extracted_patterns.json",
    model_name="Qwen/Qwen2.5-7B-Instruct"
)

# Reformulate queries
results = reformer.reformulate_queries(queries_df)
```

#### Command Line Usage

```bash
python reformer/core/reformer.py \
  --queries data/trecdl2019/reformulated_queries.tsv \
  --bm25_run data/trecdl2019/run_reformulated_queries.tsv \
  --collection data/trecdl2019/collection.tsv \
  --patterns reformer/patterns/extracted_patterns.json \
  --output results/reformer_output.tsv \
  --model Qwen/Qwen2.5-7B-Instruct
```

### Pattern Extraction

#### Python API

```python
from reformer.patterns import IterativePatternExtractor

# Extract patterns from query pairs
extractor = IterativePatternExtractor(
    data_path="data/diamond_dataset.tsv",
    model="qwen2.5:72b",
    batch_size=10
)

# Run pattern extraction
extractor.run_iterative_extraction()
```

#### Command Line Usage

```bash
python reformer/patterns/iterative_pattern_extraction.py
```

### ReFormeR API Reference

```python
class Reformulator:
    def __init__(self, bm25_run_path: str, collection_path: str, patterns_path: str, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct")
    def reformulate_query(self, query_id: str, query: str) -> Dict
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str)

class IterativePatternExtractor:
    def __init__(self, data_path: str, model: str = "qwen2.5:72b", batch_size: int = 10)
    def run_iterative_extraction(self) -> None

class PromptManager:
    def __init__(self, prompts_file: str = "reformer/prompts.json")
    def get_system_prompt(self, prompt_type: str) -> str
    def get_user_prompt(self, prompt_type: str, **kwargs) -> str
    def create_messages(self, prompt_type: str, **kwargs) -> List[Dict[str, str]]

@dataclass
class QueryPair:
    original_query: str
    reformulated_query: str
    query_id: str = ""

@dataclass  
class ReformulationPattern:
    pattern_name: str
    description: str
    transformation_rule: str
    examples: List[Tuple[str, str]]
```

## 🏗️ Baseline Methods

This repository includes several established baseline methods for comparison:

### 1. MuGI (Multi-Text Generation Integration)
- **File**: `baselines/mugi/mugi.py`
- **Approach**: Generates pseudo-documents and concatenates with original query
- **LLM Calls**: 5 (one per document)
- **Output**: `(Q × 5) + 5 pseudo-docs`

```bash
python baselines/mugi/mugi.py \
  --queries data/trecdl2019/reformulated_queries.tsv \
  --output results/mugi_output.tsv \
  --model Qwen/Qwen2.5-7B-Instruct
```

### 2. QA-EXPAND (Multi-Question Answer Generation)
- **File**: `baselines/qa_expand/qa_expand.py`
- **Approach**: Generates sub-questions, answers, and refines them
- **LLM Calls**: 3 (sub-questions → answers → refinement)
- **Output**: `(Q × 3) + refined answers`

```bash
python baselines/qa_expand/qa_expand.py \
  --queries data/trecdl2019/reformulated_queries.tsv \
  --output results/qa_expand_output.tsv \
  --env gpu
```

### 3. GenQR-Ensemble
- **File**: `baselines/genqr_ensemble/genqr_ensemble.py`
- **Approach**: Uses 10 different instruction variants for keyword generation
- **LLM Calls**: 10 (one per instruction variant)
- **Output**: `(Q × 5) + all keywords`

```bash
python baselines/genqr_ensemble/genqr_ensemble.py \
  --queries data/trecdl2019/reformulated_queries.tsv \
  --output results/genqr_output.tsv \
  --env gpu
```

### 4. FlanQR (Instruction-Based Expansion)
- **File**: `baselines/flanqr/flanqr.py`
- **Approach**: Single instruction-based expansion using Qwen
- **LLM Calls**: 1
- **Output**: `(Q × 5) + expansion text`

```bash
python baselines/flanqr/flanqr.py \
  --queries data/trecdl2019/reformulated_queries.tsv \
  --output results/flanqr_output.tsv \
  --env gpu
```

### 5. Query2Doc Variants
- **Files**: `baselines/query2doc/query2doc_*.py`
- **Approach**: Generate pseudo-documents from queries
- **Variants**: Chain-of-Thought, Few-Shot, Zero-Shot prompting

```bash
# Zero-Shot variant
python baselines/query2doc/query2doc_ZS.py \
  --target_queries data/trecdl2019/reformulated_queries.tsv \
  --output results/query2doc_zs_output.tsv

# Few-Shot variant
python baselines/query2doc/query2doc_FS.py \
  --dataset_type msmarco \
  --collection data/msmarco/collection.tsv \
  --train_queries data/msmarco/train_queries.tsv \
  --train_qrels data/msmarco/train_qrels.tsv \
  --target_queries data/trecdl2019/reformulated_queries.tsv \
  --output results/query2doc_fs_output.tsv

# Chain-of-Thought variant
python baselines/query2doc/query2doc_CoT.py \
  --target_queries data/trecdl2019/reformulated_queries.tsv \
  --output results/query2doc_cot_output.tsv
```

### Baseline API Reference

```python
class MuGIGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", num_docs: int = 5)
    def generate_pseudo_documents(self, query: str) -> List[str]
    def create_enhanced_query(self, query: str, pseudo_docs: List[str], adaptive_times: int = 5) -> str

class QAExpandGenerator:
    def __init__(self, model_name: str, num_subquestions: int = 3, env: str = "gpu")
    def generate_queries(self, queries_df: pd.DataFrame) -> pd.DataFrame

class GenQREnsembleGenerator:
    def __init__(self, model_name: str, repeat_query_weight: int = 5, env: str = "gpu")
    def generate_queries(self, queries_df: pd.DataFrame) -> pd.DataFrame

class FlanQRGenerator:
    def __init__(self, model_name: str, repeat_query_weight: int = 5, env: str = "gpu")
    def generate_queries(self, queries_df: pd.DataFrame) -> pd.DataFrame
```

## 📊 Data Requirements

### Available Datasets

This repository includes pre-processed reformulated queries for three major datasets:

- **TREC DL 2019**: `data/trecdl2019/reformulated_queries.tsv`
- **TREC DL 2020**: `data/trecdl2020/reformulated_queries.tsv`  
- **DL-Hard**: `data/dlhard/reformulated_queries.tsv`
- **Diamond Data**: `data/diamond_dataset.tsv` (Gold reformulation pairs for pattern extraction)

Each dataset includes:
- `reformulated_queries.tsv`: Query reformulations for evaluation
- `run_reformulated_queries.tsv`: TREC-formatted run files for retrieval evaluation

**Note**: The reformulated queries are pre-processed and ready for use with all baseline methods and ReFormeR. The run files can be used for retrieval evaluation and document-based pattern selection. The diamond dataset contains gold query reformulation pairs used for extracting patterns in ReFormeR.

### Dataset-Specific Examples

**TREC DL 2019**:
```bash
# ReFormeR
python reformer/core/reformer.py \
  --queries data/trecdl2019/original_queries.tsv \
  --bm25_run data/trecdl2019/run_original_queries.tsv \
  --collection data/trecdl2019/collection.tsv \
  --patterns reformer/patterns/extracted_patterns.json \
  --output results/trecdl2019_reformer.tsv

# MuGI Baseline
python baselines/mugi/mugi.py \
  --queries data/trecdl2019/reformulated_queries.tsv \
  --output results/trecdl2019_mugi.tsv
```

**TREC DL 2020**:
```bash
# ReFormeR
python reformer/core/reformer.py \
  --queries data/trecdl2020/original_queries.tsv \
  --bm25_run data/trecdl2020/run_original_queries.tsv \
  --collection data/trecdl2020/collection.tsv \
  --patterns reformer/patterns/extracted_patterns.json \
  --output results/trecdl2020_reformer.tsv

# MuGI Baseline
python baselines/mugi/mugi.py \
  --queries data/trecdl2020/reformulated_queries.tsv \
  --output results/trecdl2020_mugi.tsv
```

**DL-Hard**:
```bash
# ReFormeR
python reformer/core/reformer.py \
  --queries data/dlhard/original_queries.tsv \
  --bm25_run data/dlhard/run_original_queries.tsv \
  --collection data/dlhard/collection.tsv \
  --patterns reformer/patterns/extracted_patterns.json \
  --output results/dlhard_reformer.tsv

# MuGI Baseline
python baselines/mugi/mugi.py \
  --queries data/dlhard/reformulated_queries.tsv \
  --output results/dlhard_mugi.tsv
```

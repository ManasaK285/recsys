# EmbedBench: Cost-Aware Embedding Benchmark and Optimization Lab

**A research-inspired project for evaluating embedding dimensionality, numerical precision, semantic retrieval quality, latency, and vector storage efficiency.**

## Overview

EmbedBench is an end-to-end benchmarking framework for studying the trade-offs between embedding quality and computational efficiency in semantic retrieval systems.

The project evaluates embedding configurations with different vector dimensions and numerical precisions, then compares their retrieval performance, search latency, and estimated vector storage requirements.

The current implementation uses **`sentence-transformers/all-MiniLM-L6-v2`** as its embedding model. The project is designed to support experimentation with alternative embedding models, including Google's EmbeddingGemma, when model access is available.

The benchmark currently evaluates three embedding dimensions (384, 256, and 128) and three numerical precision configurations (float32, float16, and int8).

The goal is to identify configurations that provide a useful balance between retrieval quality, storage efficiency, and search performance.

## Research Questions

EmbedBench investigates the following questions:

1. How does reducing embedding dimensionality affect semantic retrieval quality?
2. How does numerical precision affect retrieval metrics and vector storage?
3. Can lower-dimensional embeddings preserve retrieval quality while reducing storage requirements?
4. How do embedding configurations compare in query-search latency?
5. How can retrieval quality and resource costs be combined to select an appropriate configuration?

## Features

- **Embedding generation:** Uses Sentence Transformers to generate semantic embeddings.
- **Dimensionality experiments:** Evaluates 384D, 256D, and 128D configurations.
- **Precision experiments:** Compares float32, float16, and int8 configurations.
- **Retrieval evaluation:** Computes Recall@k, Mean Reciprocal Rank (MRR), and nDCG@k.
- **Performance measurement:** Records embedding-generation time and query-search latency statistics.
- **Storage analysis:** Estimates vector storage requirements for each configuration.
- **Configuration selection:** Includes an experiment script for selecting a configuration according to the implemented optimization criteria.
- **API and dashboard:** Includes a FastAPI retrieval interface and a Streamlit dashboard.
- **Reproducibility:** Includes sample data, configuration files, experiment scripts, and tests.

## Project Structure

```text
embedbench/
├── configs/
│   └── default.yaml
├── data/
│   ├── sample_corpus.jsonl
│   └── sample_queries.jsonl
├── dashboard/
│   └── app.py
├── experiments/
│   ├── run_benchmark.py
│   └── select_configuration.py
├── outputs/
├── src/
│   └── embedbench/
│       ├── __init__.py
│       ├── models.py
│       ├── data.py
│       ├── compression.py
│       ├── indexing.py
│       ├── metrics.py
│       ├── benchmarking.py
│       ├── optimizer.py
│       ├── retrieval.py
│       └── api.py
├── tests/
│   ├── test_metrics.py
│   ├── test_compression.py
│   └── test_optimizer.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Embedding model | Sentence Transformers |
| Current benchmark model | `sentence-transformers/all-MiniLM-L6-v2` |
| Numerical computation | NumPy |
| Vector search | FAISS, where used by the indexing implementation |
| API | FastAPI |
| Interactive dashboard | Streamlit |
| Testing | pytest |
| Configuration | YAML |
| Experimental data | JSONL and generated result files |

## Installation

The following commands use Windows PowerShell.

### 1. Navigate to the project

```powershell
cd "C:\Users\Manasa\Desktop\MK\Personal\mk\Google2\G2\embedbench"
```

### 2. Create and activate a virtual environment

```powershell
py -3.14 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If you have a different supported Python version installed, replace `-3.14` with that version.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The first benchmark run may download the selected embedding model from Hugging Face. An internet connection is required unless the model is already cached.

## Running the Benchmark

Run the benchmark with the publicly accessible MiniLM model:

```powershell
python experiments\run_benchmark.py --model sentence-transformers/all-MiniLM-L6-v2 --dimensions 384 256 128 --device cpu
```

The benchmark evaluates the requested dimensions across the supported precision configurations and reports retrieval metrics, timing measurements, and vector storage estimates.

The current sample experiment uses:

- **Corpus size:** 10 documents
- **Query count:** 5 queries
- **Embedding dimensions:** 384, 256, 128
- **Precision configurations:** float32, float16, int8
- **Device:** CPU

The script saves its results in the `outputs` directory.

Inspect the generated files with:

```powershell
Get-ChildItem .\outputs
```

### Run configuration selection

```powershell
python experiments\select_configuration.py
```

This script runs the project's configuration-selection workflow. The selection should be interpreted according to the objective and constraints implemented in the optimizer, rather than assumed to represent a universally optimal embedding configuration.

### Run tests

```powershell
python -m pytest
```

### Launch the dashboard

```powershell
python -m streamlit run dashboard\app.py
```

### Launch the API

If the API module exposes the FastAPI application as `app` in `embedbench.api`, run:

```powershell
python -m uvicorn embedbench.api:app --reload
```

Use the API's documented routes to submit retrieval requests. If the module or application variable has a different name in your local implementation, adjust the Uvicorn target accordingly.

## Evaluation Metrics

### Recall@k

Measures the proportion of relevant documents retrieved within the top-k results.

Higher values indicate that more relevant documents are being retrieved.

### Mean Reciprocal Rank (MRR)

Measures the reciprocal rank of the first relevant result, averaged across queries.

Higher MRR indicates that relevant results tend to appear closer to the top.

### Normalized Discounted Cumulative Gain (nDCG@k)

Evaluates ranking quality while accounting for the positions of relevant results. Relevant results appearing near the top contribute more to the score.

### Embedding-generation time

Measures the elapsed time associated with embedding generation during the benchmark. This should be distinguished from query-search latency.

### Query-search latency

Measures the time taken to search for results for individual queries. The benchmark reports mean and p95 latency fields; the p95 implementation should be validated before making tail-latency claims.

### Vector storage

Estimates the memory required to store embedding vectors for the corpus. It does not necessarily include index metadata, document text, process overhead, or other application storage.

## Experimental Results

The following results were obtained using `sentence-transformers/all-MiniLM-L6-v2` on the bundled sample dataset.

| Dimension | Precision | Recall@k | MRR | nDCG@k | Vector storage |
|---:|---|---:|---:|---:|---:|
| 384 | float32 | 0.80 | 0.8500 | 0.7455 | 15,360 bytes |
| 384 | float16 | 0.80 | 0.8500 | 0.7455 | 7,680 bytes |
| 384 | int8 | 0.90 | 0.8500 | 0.7930 | 3,840 bytes |
| 256 | float32 | 0.90 | 0.8667 | 0.8153 | 10,240 bytes |
| 256 | float16 | 0.90 | 0.8667 | 0.8153 | 5,120 bytes |
| 256 | int8 | 0.90 | 0.8667 | 0.8153 | 2,560 bytes |
| 128 | float32 | 0.90 | 0.8667 | **0.8314** | 5,120 bytes |
| 128 | float16 | 0.90 | 0.8667 | **0.8314** | 2,560 bytes |
| 128 | int8 | 0.90 | 0.8667 | **0.8314** | **1,280 bytes** |

*Storage figures are the values reported by the implementation. They represent estimated vector storage, not total system memory.*

### Interpretation

**1. The 128D configurations achieved the highest observed nDCG@k.**

All three 128D precision configurations achieved an nDCG@k of approximately 0.8314, higher than the 384D float32 baseline of approximately 0.7455.

This result shows that, on the bundled sample dataset, reducing the embedding dimension did not reduce measured ranking quality.

**2. The 128D int8 configuration was the most storage-efficient.**

The 128D int8 configuration used 1,280 bytes for the reported corpus vectors, compared with 15,360 bytes for 384D float32.

That corresponds to an observed storage reduction of approximately 91.7%, while the measured Recall@k, MRR, and nDCG@k were higher in this particular comparison.

**3. Reduced precision did not change every ranking.**

For the 256D and 128D configurations, all three precision settings produced identical reported retrieval metrics. This is consistent with the rankings remaining unchanged across these settings on the evaluated queries.

However, the implementation should be checked to ensure that the compressed representations are actually used during search, rather than being used only for storage estimation.

**4. Smaller vectors are not guaranteed to be faster.**

The recorded query-search times varied across configurations and were very small. On a corpus of only 10 documents, measurement noise and implementation overhead can dominate these values. The results are insufficient to establish a reliable latency winner.

## Current Best Configuration

For the current experiment, **128D int8 is the provisional choice when minimizing vector storage is the main objective**.

It achieved:

- Recall@k: 0.90
- MRR: 0.8667
- nDCG@k: 0.8314
- Estimated vector storage: 1,280 bytes

This is a result for the current model, data, and implementation—not proof that 128D int8 is optimal for other datasets or production systems.

## EmbeddingGemma Support

The project was initially motivated by research into efficient embedding models, including EmbeddingGemma.

The current successful experiment uses MiniLM because the Hugging Face repository for `google/embeddinggemma-300m` requires authorized access.

To benchmark EmbeddingGemma:

1. Visit `https://huggingface.co/google/embeddinggemma-300m`.
2. Complete any required access request or acceptance steps.
3. Authenticate with Hugging Face using the CLI supported by your installed version.
4. Run the benchmark with the EmbeddingGemma model identifier.

For example:

```powershell
python -m pip install --upgrade huggingface_hub
hf auth login
python experiments\run_benchmark.py --model google/embeddinggemma-300m --dimensions 384 256 128 --device cpu
```

The model must be accessible to your account. The command assumes the installed Sentence Transformers version can load the selected model and that the requested dimensions are supported by the implementation. Verify both before comparing results.

Results from different models should be reported separately because model architecture, training data, embedding behavior, and supported dimensions can affect retrieval quality.

## Limitations

The current benchmark is an initial experimental prototype, with several limitations.

- **Small evaluation set:** Ten corpus documents and five queries are insufficient to establish robust generalization.
- **Model coverage:** The reported results use MiniLM, not EmbeddingGemma.
- **Quantization validation:** Confirm that compressed embeddings are used in retrieval before attributing metric changes to quantization.
- **Latency measurement:** Five queries do not provide a robust p95 latency estimate. Repeat experiments and use a larger query set.
- **Storage scope:** Estimated vector storage excludes some index and application overhead.
- **Relevance labels:** Results depend on the relevance labels supplied with the sample data.
- **Hardware dependence:** Timing results depend on hardware, runtime, caching, and software versions.
- **Configuration dependence:** The best dimension and precision may change with the dataset, model, relevance distribution, and retrieval workload.

## Future Work

Potential extensions include:

1. Evaluate on larger public semantic retrieval datasets with explicit relevance judgments.
2. Compare EmbeddingGemma, MiniLM, and other embedding models using the same evaluation protocol.
3. Validate quantization and dimensionality reduction against actual search-time representations.
4. Measure p50, p95, and p99 query latency over repeated runs.
5. Track peak memory use, total index size, throughput, and embedding-generation cost.
6. Use multiple random seeds or bootstrap confidence intervals to estimate metric uncertainty.
7. Introduce a configurable multi-objective selection method that balances retrieval quality, storage, and latency.
8. Add experiment tracking and export reproducible reports.

## Conclusion

EmbedBench provides a practical foundation for benchmarking semantic retrieval configurations and investigating the trade-offs among embedding dimensionality, numerical precision, retrieval quality, and storage.

In the current MiniLM experiment, 128D int8 achieved the best observed nDCG@k and the smallest estimated vector storage. These findings motivate larger, more rigorous experiments to determine whether the same trade-offs hold across models, datasets, and retrieval workloads.

The project is intended as a reproducible research prototype rather than a production-ready benchmark or a definitive evaluation of EmbeddingGemma.

## License and Attribution

This project is an independent research-inspired implementation. Cite the relevant embedding-model papers and model documentation when using it in research or derivative work. Confirm the license terms of each model and dependency before redistribution or commercial use.

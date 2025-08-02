## Section 7 – Quantization in Vector Databases

### 7.1 Intro

Quantization is a method that reduces the size of embedding vectors to save memory and accelerate search — especially useful when working with millions or billions of vectors.

It’s a classic performance technique: **compress the data**, run fast lookups, and recover approximate results with minimal loss.

---

### 7.2 What is Quantization?

Quantization turns high-precision vectors (e.g., 32-bit floats) into **low-bit approximations** (e.g., 8-bit integers or codebook lookups). These smaller representations are faster to store, load, and compare.

---

### 📦 Analogy

> Think of storing color images. Instead of 16 million RGB colors (float precision), you can use a **palette of 256 colors** (int8), and just store the index of each color in the palette.
> It’s smaller, faster — and if done right, you barely notice the difference.

---

### 7.3 Why Use Quantization?

| Situation                       | Benefit                                              |
| ------------------------------- | ---------------------------------------------------- |
| Large datasets (10M–1B vectors) | Shrinks memory usage 4× to 16×                       |
| Low-latency search needed       | Enables fast SIMD/GPU-based comparisons              |
| Budget-constrained systems      | Reduces CPU cache misses and I/O volume              |
| Deploying on edge/mobile        | Makes LLM-powered tools usable offline or in-browser |

---

### 7.4 How It Works

There are two popular types:

#### A. **Scalar Quantization (int8)**

Each float is linearly scaled to an 8-bit integer.

```text
[0.12, 0.98, -0.44] → [15, 240, 64]
```

Pros: Simple, fast, easy to enable.
Cons: Less precise with small floats or large ranges.

---

#### B. **Product Quantization (PQ)**

The vector is split into sub-vectors. Each sub-vector is **matched to a codebook**, and only the code index is stored.

```text
[....] → [3, 145, 12, 88]  # indices into small codebooks
```

Pros: Excellent compression ratio
Cons: Slightly slower to search, requires fine-tuning codebooks

---

### 7.5 Trade-offs

| Metric            | With Quantization                   | Without                           |
| ----------------- | ----------------------------------- | --------------------------------- |
| Memory usage      | Low (e.g., 1–2 GB for 100M vectors) | High (e.g., 10–16 GB)             |
| Speed             | Fast SIMD / GPU                     | Slower, esp. for disk/large index |
| Accuracy (recall) | Slightly reduced (\~1–3%)           | High                              |
| Latency           | Sub-millisecond                     | 5–30ms typical                    |

---

### 7.6 How to Enable in Qdrant

Quantization is enabled per collection at creation time using `quantization_config`.

#### A. Enabling **int8** quantization

```python
from qdrant_client.models import ScalarQuantization

client.create_collection(
    collection_name="compressed_vectors",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantization.Config(
            type="int8",
            quantile=0.99,           # outlier trimming
            always_ram=True          # load in RAM for performance
        )
    )
)
```

#### B. Enabling **Product Quantization (PQ)** (if supported by current backend)

```python
from qdrant_client.models import ProductQuantization

client.create_collection(
    collection_name="pq_vectors",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ProductQuantization(
        product=ProductQuantization.Config(
            compression="x4",       # 4× compression
            always_ram=False        # or True
        )
    )
)
```

---

### 7.7 When Not to Use Quantization

* On small datasets (< 1M vectors) where RAM is not an issue
* If full accuracy (recall = 100%) is mission-critical
* When doing re-ranking with LLMs, quantization is okay because recall loss can be recovered downstream

---

### 7.8 Takeaways

* Quantization drastically reduces memory and improves search speed
* Two main types: **int8 (scalar)** and **PQ (product)**
* Trade-off: slightly lower recall for faster performance and lower cost
* Use it when **scaling**, **cost**, or **latency** are priorities
* Works great in combination with **HNSW indexing** in Qdrant

## Section 8 – Multimodal Embeddings in Vector Databases

### 8.1 Intro

Most vector DBs, including Qdrant, are not limited to **text**. They can store and index any data that can be converted into **embeddings** — fixed-size numeric vectors.

Multimodal search refers to working across **text, images, audio, code, video**, and even structured data — all within the same framework.

---

### 8.2 What are Multimodal Embeddings?

Multimodal embeddings are vector representations of different data types that are mapped into a **shared or aligned vector space**.

This makes it possible to:

* Search an image using a text query
* Find relevant audio clips by uploading a voice prompt
* Search code using natural language
* Match video scenes to a written description

---

### 📦 Analogy

> Imagine every image, sound, or document being **translated into a GPS coordinate**.
> If two things are “close in meaning”, their vectors will be **close in space** — no matter if one was a sentence and the other was a video.

---

### 8.3 How Images, Audio, Code Get Embedded

Each modality uses a model trained to convert input data into a fixed-length vector. Here’s how each works:

---

#### A. **Images → Vector**

| Step    | How                                            |
| ------- | ---------------------------------------------- |
| Input   | `.jpg`, `.png`, etc.                           |
| Model   | CLIP, ViT (Vision Transformer), DINOv2, ResNet |
| Output  | 512–1024 dimensional float vector              |
| Example | `vector = image_encoder(image)`                |

Popular tools:

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

img = Image.open("dog.jpg")
inputs = processor(images=img, return_tensors="pt")
outputs = model.get_image_features(**inputs)
vector = outputs[0].detach().numpy()
```

---

#### B. **Audio → Vector**

| Step     | How                                                |
| -------- | -------------------------------------------------- |
| Input    | `.mp3`, `.wav`                                     |
| Model    | Whisper, Wav2Vec2, OpenL3, CLAP                    |
| Output   | 512–768 dimensions                                 |
| Use case | “Find clips with guitar solos” via text or humming |

---

#### C. **Code → Vector**

| Step     | How                                                           |
| -------- | ------------------------------------------------------------- |
| Input    | `.py`, `.js`, `.cpp`                                          |
| Model    | CodeBERT, CodeT5, StarCoder                                   |
| Output   | 768–1024 dimensions                                           |
| Use case | "Find functions that parse JSON files" (via natural language) |

---

#### D. **Text → Vector**

Already covered earlier — via models like MiniLM, Ada, or BGE.

---

### 8.4 How Multimodal Search Works

#### Use case: Text-to-Image

> Search for “a dog on a skateboard” and return matching images.

**How it works:**

1. The text is encoded with a **text encoder** (like CLIP’s text head)
2. All images are embedded with the **image encoder**
3. A vector DB stores the image embeddings
4. The text query vector is compared against the stored image vectors
5. Cosine similarity gives the ranked results

#### General pattern:

```
Query (text/audio/image/code)
   ↓
Embedding model → query vector
   ↓
Vector DB → ANN Search
   ↓
Return matching items from the desired modality
```

---

### 8.5 Example: Image Search by Text

```python
# Step 1: Embed the text query
text = "a cat sitting on a keyboard"
text_inputs = processor(text=[text], return_tensors="pt")
text_vector = model.get_text_features(**text_inputs)[0].detach().numpy()

# Step 2: Search in vector DB
results = client.search(
    collection_name="image_embeddings",
    query_vector=text_vector.tolist(),
    limit=3
)

# Step 3: Results = image IDs or filepaths stored as payload
for hit in results:
    print(f"Score={hit.score:.3f}, Image path: {hit.payload['file']}")
```

---

### 8.6 Use Cases

| Modality      | Example                                                      |
| ------------- | ------------------------------------------------------------ |
| Text ↔ Image  | Search art catalog by prompt (“sunset over forest”)          |
| Audio ↔ Text  | Search sound effects by description (“explosion underwater”) |
| Text ↔ Code   | Natural language search over codebase                        |
| Image ↔ Image | Find similar frames in a video                               |
| Text ↔ Video  | “Find clips where someone is pouring tea”                    |

---

### 8.7 How to Store These in Vector DBs

Each data item becomes:

```json
{
  "id": 1,
  "vector": [0.123, 0.875, ...],
  "payload": {
    "file": "images/cat.png",
    "type": "image",
    "label": "cat on keyboard"
  }
}
```

Payloads help filter (e.g., `type == "image"`), while the vector is used for similarity.

---

### 8.8 Key Takeaways

* Multimodal search works by converting all data types into embeddings
* Text, image, audio, and code can live side-by-side in a vector DB
* Querying can be cross-modal: e.g., search images using text
* Cosine similarity ranks results regardless of original data type
* Payload metadata enables filtering per modality, category, etc.
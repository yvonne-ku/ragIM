import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from server import settings


"""
 - buffer_size
 - threshold_percentile
"""
def semantic_split(input_path, output_path, buffer_size=1, threshold_percentile=90):
    """
    Semantic Baseline: Sliding Window Semantic Splitter
    Maintains two windows (left and right) of size window_size around each possible split point.
    Calculates the similarity between the average embeddings of these two windows.
    Identifies 'dips' in similarity as topic split points.

    - buffer_size: radius of the window
    - threshold_percentile
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get("content", [])
    if not messages:
        return

    # 1. Initialize Embeddings
    print(f"Loading model: BAAI/bge-m3...")
    model = SentenceTransformer('BAAI/bge-m3')

    # 2. Build windows and get embeddings
    print("Encoding windows...")
    window_sentences = []
    for i in range(len(messages)):
        start = max(0, i - buffer_size)
        end = min(len(messages), i + buffer_size + 1)
        window_text = " ".join([m.get("text", "") for m in messages[start:end]])
        window_sentences.append(window_text)
    embeddings = model.encode(window_sentences, show_progress_bar=True)

    # 3. Calculate similarities of adjacent windows
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = util.cos_sim(embeddings[i], embeddings[i + 1])
        similarities.append(similarity.item())

    # 4. Identify dips (local minima)
    low_percentile = 100 - threshold_percentile
    breakpoint_threshold = np.percentile(similarities, low_percentile)
    print(f"Calculated breakpoint threshold: {breakpoint_threshold:.4f}")

    # 5. Split
    chunks = []
    current_chunk = [messages[0]]
    for i in range(len(similarities)):
        if similarities[i] < breakpoint_threshold:
            chunks.append(current_chunk)
            current_chunk = [messages[i + 1]]
        else:
            current_chunk.append(messages[i + 1])
    if current_chunk:
        chunks.append(current_chunk)

    # 6. Save results
    # Format chunks with chunk_id
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i + 1:05d}"
        formatted_chunks.append({
            "chunk_id": chunk_id,
            "messages": chunk
        })

    result = {
        "window_id": data.get("window_id", "2"),
        "method": "semantic_baseline",
        "buffer_size": buffer_size,
        "threshold_percentile": float(threshold_percentile),
        "chunks": formatted_chunks
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_path = os.path.join(output_path, f"ibm_semantic_split_{buffer_size}_{threshold_percentile}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Semantic Baseline: Processed {len(messages)} messages into {len(chunks)} chunks.")
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    input_file = os.path.join(settings.basic_settings.RAW_JSON_PATH, "ibm_all.json")
    output_path = os.path.join(settings.basic_settings.CHUNKS_PATH)

    semantic_split(input_file, output_path)

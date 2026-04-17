import json
import os

def naive_split(input_path, output_path, chunk_size=500):
    """
    Naive Baseline: Fixed-length splitting (RecursiveCharacterTextSplitter style).

    - chunk_size
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get("content", [])
    chunks = []
    current_chunk = []
    current_length = 0

    for msg in messages:
        msg_text = msg.get("text", "")
        msg_len = len(msg_text)

        if current_length + msg_len > chunk_size and current_chunk:
            # Save current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = [msg]
            current_length = msg_len
        else:
            current_chunk.append(msg)
            current_length += msg_len

    if current_chunk:
        chunks.append(current_chunk)

    # Save as a list of chunks, where each chunk is a list of messages
    result = {
        "window_id": data.get("window_id", "1"),
        "method": "naive_baseline",
        "chunk_size": chunk_size,
        "chunks": chunks
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Naive Baseline: Processed {len(messages)} messages into {len(chunks)} chunks.")
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    # Base paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir)) # ragIM
    input_file = os.path.join(project_root, "data", "knowledge_base", "samples", "content", "ubuntu_all.json")
    output_file = os.path.join(current_dir, "ubuntu_naive_split.json")

    naive_split(input_file, output_file)

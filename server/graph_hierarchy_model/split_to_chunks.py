import json
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """精确计算单条消息文本的 token 数"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

"""
 - max_chunk_size
 - overlap
"""
def split_into_chunks(json_file_path: str, output_path: str):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    window_id = data.get("window_id", "unknown")
    messages = data.get("content", [])


    max_chunk_size = 600      # 建议 500~800
    overlap = 1         # 与前一个 chunk 重叠的消息条数
    model = "gpt-3.5-turbo"

    # 分块
    chunks = []
    current_chunk = []
    current_tokens = 0
    for msg in messages:
        msg_text = msg.get("text", "")
        msg_tokens = count_tokens(msg_text, model)    # 只会本地计算，不会调用 API

        if current_tokens + msg_tokens > max_chunk_size and current_chunk:
            chunks.append(current_chunk)

            # 重叠处理：新 chunk 从前一个 chunk 的最后 overlap 条消息开始
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:]
            current_tokens = sum(count_tokens(m.get("text", ""), model) for m in current_chunk)

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

   # 输出 json
    output = {
        "window_id": window_id,
        "method": "graph_hierarchy_split",
        "max_chunk_size": max_chunk_size,
        "overlap": overlap,
        "chunks": []
    }
    for idx, msg_list in enumerate(chunks, start=1):
        chunk_obj = {
            "chunk_id": f"chunk_{idx:05d}",
            "messages": msg_list  # 直接保留原始消息对象，包含 msg_id, from, to, text, topic_id
        }
        output["chunks"].append(chunk_obj)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"已处理窗口 {window_id}，生成 {len(chunks)} 个 chunk，保存至 {output_path}")


if __name__ == "__main__":
    input_file = "D:\\MyProjects\\ragIM\\data\\raw_json_data\\ibm_all.json"
    output_file = "D:\\MyProjects\\ragIM\\data\\processed_chunks\\ibm_graph_hierarchy_split.json"
    split_into_chunks(input_file, output_file)
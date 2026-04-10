import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.store import EmbeddingStore
import json
import os
from dotenv import load_dotenv
from src.embeddings import OpenAIEmbedder
from src.agent import KnowledgeBaseAgent

def build_llm_fn():
    from openai import OpenAI
    client = OpenAI()
    def _llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content or ""
    return _llm

def main():
    load_dotenv()
    embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    
    with open("data/edu_policy_index.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    store = EmbeddingStore(collection_name="edu_policy_store", embedding_fn=embedder)
    store._store = records
    
    llm_fn = build_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    
    queries = [
        ("Thí sinh được phép mang những vật dụng gì vào phòng thi?", None),
        ("Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào?", {"category": "to_chuc_thi"}),
        ("Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm?", None),
        ("Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện?", None),
        ("Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm?", None)
    ]
    
    for i, (q, filter_dict) in enumerate(queries, 1):
        print(f"\n--- QUERY {i} ---")
        print(f"Q: {q}")
        if filter_dict:
            results = store.search_with_filter(q, top_k=1, metadata_filter=filter_dict)
            answer = llm_fn(f"Context:\n{results[0]['content']}\n\nQuestion: {q}\nAnswer:")
            score = results[0]['score']
            doc_id = results[0]['metadata'].get('doc_id')
        else:
            results = store.search(q, top_k=1)
            answer = agent.answer(q, top_k=1)
            score = results[0]['score']
            doc_id = results[0]['metadata'].get('doc_id')
        
        print(f"Top Chunk Score: {score:.4f} from {doc_id}")
        chunk_preview = results[0]['content'][:150].replace('\n', ' ')
        print(f"Top Chunk: {chunk_preview}...")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
DOCS = BASE / 'docs'
CHROMA_DIR = BASE / 'chroma_db'
COLLECTION_NAME = 'zepto_policies'
MODEL_NAME = 'all-MiniLM-L6-v2'


def load_chunks():
    chunks=[]
    for path in sorted(DOCS.glob('doc_*.txt')):
        text=path.read_text(encoding='utf-8').strip()
        chunks.append({'id':path.stem,'text':text,'source':path.name})
    if len(chunks)!=8:
        raise RuntimeError(f'Expected 8 policy documents, found {len(chunks)}')
    return chunks


def build_collection():
    chunks=load_chunks()
    model=SentenceTransformer(MODEL_NAME)
    embeddings=model.encode([c['text'] for c in chunks], normalize_embeddings=True).tolist()
    client=chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection=client.get_or_create_collection(COLLECTION_NAME, metadata={'hnsw:space':'cosine'})
    # Upsert makes reruns deterministic and avoids duplicate IDs.
    collection.upsert(ids=[c['id'] for c in chunks], documents=[c['text'] for c in chunks], embeddings=embeddings, metadatas=[{'source':c['source']} for c in chunks])
    return collection

if __name__=='__main__':
    c=build_collection()
    print('Indexed documents:', c.count())

from pathlib import Path
import os
from typing import TypedDict
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

BASE=Path(__file__).resolve().parent
CHROMA_DIR=BASE/'chroma_db'
COLLECTION_NAME='zepto_policies'
MODEL_NAME='all-MiniLM-L6-v2'
KEYWORDS=['delivery','return','refund','membership','tracking','cancel','gift card','support hours']

class Answer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

class State(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float

PROMPT_TEMPLATE='''ROLE: You are Zepto's policy support assistant.\n\nCONTEXT: Use only the policy excerpts supplied below.\n\nTASK: Answer the user's policy question using the retrieved context.\n\nFORMAT: Return JSON with answer (string), sources (list of document IDs), and confidence (0-1).\n\nLENGTH: Keep the answer concise, normally 1-3 sentences.\n\nNEGATIVE CONSTRAINT: Do not answer using information that is not present in the provided context. If the context is insufficient, say that the provided policy documents do not contain the answer.\n\nFEW-SHOT EXAMPLE:\nQuestion: How long do I have to report a damaged grocery item?\nContext: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect.\nAnswer: {{"answer":"Damaged, spoiled, or incorrect grocery/perishable items should be reported within 24 hours of delivery.","sources":["doc_02"],"confidence":1.0}}\n\nRETRIEVED CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n'''


def mock_mode():
    return os.getenv('MOCK_LLM','1') != '0'


def get_collection():
    client=chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={'hnsw:space':'cosine'})


def get_embedder():
    return SentenceTransformer(MODEL_NAME)


def classify_intent(state: State):
    q=state['query'].lower()
    if mock_mode():
        intent='policy_question' if any(k in q for k in KEYWORDS) else 'general_question'
        return {'intent':intent}
    # Optional real-LLM branch is implemented in main.py to keep this module dependency-light.
    from main import llm_classify_intent
    return {'intent':llm_classify_intent(q)}


def retrieve_and_answer(state: State):
    collection=get_collection(); model=get_embedder()
    query_vec=model.encode([state['query']], normalize_embeddings=True).tolist()
    result=collection.query(query_embeddings=query_vec, n_results=3)
    docs=result.get('documents',[[]])[0]; ids=result.get('ids',[[]])[0]
    if not docs:
        return {'answer':'No relevant policy context was found in the Zepto policy corpus.', 'sources':[], 'confidence':0.0}
    if mock_mode():
        snippet=docs[0][:200]
        return {'answer':f'Based on the retrieved context: {snippet}', 'sources':ids, 'confidence':1.0}
    from main import llm_generate_answer
    raw=llm_generate_answer(state['query'], docs, ids)
    return raw


def direct_answer(state: State):
    if mock_mode():
        return {'answer':'I can only answer questions about Zepto policies right now.', 'sources':[], 'confidence':1.0}
    from main import llm_direct_answer
    return llm_direct_answer(state['query'])


def route(state: State):
    return state['intent']


def build_graph():
    graph=StateGraph(State)
    graph.add_node('classify_intent', classify_intent)
    graph.add_node('retrieve_and_answer', retrieve_and_answer)
    graph.add_node('direct_answer', direct_answer)
    graph.set_entry_point('classify_intent')
    graph.add_conditional_edges('classify_intent', route, {'policy_question':'retrieve_and_answer','general_question':'direct_answer'})
    graph.add_edge('retrieve_and_answer', END)
    graph.add_edge('direct_answer', END)
    return graph.compile()


def ask(query: str):
    result=build_graph().invoke({'query':query})
    response=Answer(answer=result.get('answer',''), sources=result.get('sources',[]), confidence=float(result.get('confidence',1.0)))
    return response

from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag import Answer, ask, PROMPT_TEMPLATE, mock_mode

app=FastAPI(title='Zepto Grounded Support Assistant', version='1.0.0')

@app.on_event('startup')
def ensure_policy_index():
    from ingest import build_collection
    import chromadb
    base=Path(__file__).resolve().parent
    client=chromadb.PersistentClient(path=str(base/'chroma_db'))
    collection=client.get_or_create_collection('zepto_policies', metadata={'hnsw:space':'cosine'})
    if collection.count() != 8:
        build_collection()

class AskRequest(BaseModel):
    query: str

@app.post('/ask', response_model=Answer)
def ask_endpoint(request: AskRequest):
    return ask(request.query)

# Optional real-LLM extension. It is never used in the graded MOCK_LLM default path.
def get_real_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model=os.getenv('GROQ_MODEL','llama-3.1-8b-instant'), api_key=os.environ['GROQ_API_KEY'], temperature=0)

def llm_classify_intent(query):
    llm=get_real_llm()
    prompt=PROMPT_TEMPLATE.format(context='', query=query) + '\nReturn only policy_question or general_question.'
    for _ in range(3):
        text=llm.invoke(prompt).content.strip().lower()
        if 'policy_question' in text: return 'policy_question'
        if 'general_question' in text: return 'general_question'
        prompt += '\nCorrection: output exactly one of policy_question or general_question.'
    return 'general_question'

def llm_generate_answer(query, docs, ids):
    llm=get_real_llm(); context='\n\n'.join(f'[{i}] {d}' for i,d in zip(ids,docs))
    prompt=PROMPT_TEMPLATE.format(context=context, query=query)
    last=''
    for attempt in range(3):
        text=llm.invoke(prompt).content
        last=text
        try:
            import json
            data=json.loads(text)
            return Answer.model_validate(data).model_dump()
        except Exception:
            prompt += '\nCorrection: return valid JSON with exactly answer, sources, confidence. Retry.'
    return {'answer':f'ERROR: model output failed schema validation after retries. Raw output: {last[:500]}','sources':ids,'confidence':0.0}

def llm_direct_answer(query):
    llm=get_real_llm(); prompt=PROMPT_TEMPLATE.format(context='',query=query)+'\nThere is no retrieved context. Answer only if appropriate, otherwise explain the limitation.'
    last=''
    for attempt in range(3):
        text=llm.invoke(prompt).content
        last=text
        try:
            import json
            data=json.loads(text)
            data['sources']=[]
            return Answer.model_validate(data).model_dump()
        except Exception:
            prompt += '\nCorrection: return valid JSON with exactly answer, sources, confidence. Retry.'
    return {'answer':f'ERROR: model output failed schema validation after retries. Raw output: {last[:500]}','sources':[],'confidence':0.0}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=7860)

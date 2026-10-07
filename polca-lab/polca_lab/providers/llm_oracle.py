from __future__ import annotations
import json, os, urllib.request

class OpenAICompatibleOracle:
    """Optional real-LLM proposal oracle using any OpenAI-compatible endpoint."""
    def __init__(self, model=None, base_url=None, api_key=None):
        self.model=model or os.getenv('POLCA_MODEL','gpt-4.1-mini')
        self.base_url=(base_url or os.getenv('POLCA_BASE_URL','https://api.openai.com/v1')).rstrip('/')
        self.api_key=api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key: raise RuntimeError('Set OPENAI_API_KEY before using --oracle llm')
    def propose(self,parent,feedback,summary,n=3):
        prompt=f"""You are a stochastic generative optimizer. Improve the candidate system policy.
Return ONLY valid JSON: {{\"candidates\":[\"...\"]}} with exactly {n} materially different candidates.
CURRENT:\n{parent}\nLOCAL FEEDBACK:\n{feedback}\nHISTORY SUMMARY:\n{summary}\nKeep useful behavior and make concrete improvements."""
        body=json.dumps({'model':self.model,'temperature':0.8,'messages':[{'role':'system','content':'You optimize policies.'},{'role':'user','content':prompt}]}).encode()
        req=urllib.request.Request(self.base_url+'/chat/completions',data=body,headers={'Authorization':f'Bearer {self.api_key}','Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=90) as r: data=json.loads(r.read())
        content=data['choices'][0]['message']['content'].strip()
        if content.startswith('```'): content=content.split('```')[1].replace('json','',1).strip()
        out=json.loads(content).get('candidates',[])
        return [str(x) for x in out[:n]]

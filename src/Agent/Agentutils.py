from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langgraph.graph import START , END , StateGraph 
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import os
import textwrap

from typing import TypedDict , Sequence , Annotated 

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL",'http://localhost:11434')
print(OLLAMA_URL)

_MODEL_NAME : str = os.getenv('MODEL_NAME','qwen3.5:0.8b')

Summarizer_Model =  ChatOllama(
    model= _MODEL_NAME,
    reasoning=False,
    temperature=.6,
    base_url=OLLAMA_URL
)

def reformat_docstring(docstring):
    if not docstring:
        return ""
    return textwrap.dedent(docstring.expandtabs()).strip()

class AgentState(TypedDict):
    user_input : str 
    query : str 
    Msg : Annotated[Sequence[BaseMessage],add_messages]
    research_notes: Sequence[str]
    solution: str
    is_sufficient: bool
    



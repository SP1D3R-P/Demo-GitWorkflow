
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import pathlib

import logging
from Agent import _VectDB , Agentutils
# from uuid import uuid4

import pickle
import os 

DB_PATH = pathlib.Path(os.getenv('DB_PATH','./data'))
DB_PATH.mkdir(exist_ok=True,parents=True)

DB = _VectDB.VDBObject()
DB_Collection = DB.GetCollection('PdfDataBase')

LOADED_FILES_PATH = DB_PATH / "LoadedFiles.pkl"

if LOADED_FILES_PATH.exists() :
    with LOADED_FILES_PATH.open('rb') as fp :
        LoadedFiles = pickle.load(fp)
else :
    LoadedFiles = set()


def dumpData():
    global LoadedFiles 
    with LOADED_FILES_PATH.open('wb') as fp :
        pickle.dumps(
            LoadedFiles,
            fp
        )


def AddFile(path : pathlib.Path , * ,  chunk_size : int = 500 , overlap : int = 50 ) -> bool :
    global LoadedFiles
    
    try :
        if not path.exists() :
            raise FileNotFoundError(
                f'Not Found {path}'
            )
        
        if path in LoadedFiles :
            return True
        
        blobs = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',' '],
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        ).split_text(
            '\n'.join(_VectDB.VDBObject.GetPdfStr(path)) # This is kinda bad idea
        )

        DB_Collection.upsert(
            ids = [ f'{path.name}-{i}' for i in range(len(blobs)) ], # I think Using UUID4 is better but then hard to grab ids
            documents=blobs,
            metadatas=[{'file-name' : str(path.resolve()) } for _ in range(len(blobs))]
        )
        LoadedFiles.add(path)
        dumpData()
        return True
    
    except Exception as e : 
        logging.error(f"Error Occured During Adding File {path} Reason :: {e}")
        return False


ResearcherPrompt = PromptTemplate.from_template(
    template= Agentutils.reformat_docstring(
        """
        Rewrite the following user issue for optimal vector search. Make it precise without adding any extra information
        Focus on technical keywords and intent: {user_input}
        """
    )
)

def Researcher(State : Agentutils.AgentState ):
    """Searches the Vector DB for specific facts"""
    
    global ResearcherPrompt

    result : AIMessage= Agentutils.Summarizer_Model.invoke(
        ResearcherPrompt.format(
            user_input=State['user_input']
        )
    )

    qresult = DB_Collection.query(
        query_texts=[
            result.content
        ],
        n_results=15
    )

    return {'query':result.content,'research_notes':qresult['documents'][0]}

    
    
    
import chromadb
import numpy as np
import pathlib
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader 
from typing import Sequence , Dict , Optional , Iterator
import os 
import uuid
from ollama import embed

from Agent import Agentutils

class VDBObject : 

    def __init__(
                self,*,
                path : pathlib.Path = pathlib.Path(os.getenv('DB_PATH','./DB')),
                embeddings : str = os.getenv('EMBEDDING_MODEL','embeddinggemma:300m')
            ):
        self.Client = chromadb.PersistentClient(
            path
        )
        self.EmbeddingModel = embeddings
        old_collections = self.Client.list_collections()
        self.Collections = {collection.name : collection for collection in old_collections}
    
    @staticmethod
    def GetEmbb(text : str,*,truncate : bool =True,dim : int = 512 ) -> np.ndarray :
        # This is Used For testing only [NOT Used in Vector Embedding]
        return np.array(
            embed(
                model='embeddinggemma:300m',
                input=text,
                truncate=truncate,
                dimensions=dim
            ).embeddings[0]
        )
    
    def GetCollection(self,collection_name : str , * , shouldCreate = True ) -> chromadb.Collection: 
        """
        GetCollections 

        Args:
            collection_name (str) : name of the collections 
        Returns:
            chromadb.Collection
        """
        if shouldCreate :
            return self.Client.get_or_create_collection(
                collection_name,
                embedding_function=OllamaEmbeddingFunction(
                    model_name=self.EmbeddingModel,
                    url= Agentutils.OLLAMA_URL
                )
            )
        return self.Client.get_collection(
            collection_name,
            embedding_function= OllamaEmbeddingFunction(
                    model_name=self.EmbeddingModel,
                    url=Agentutils.OLLAMA_URL
                )
        ) # This Can Raise Value Error if not Present 
    
    def AddData(
            self,
            collection_name : str ,
            docs : Sequence[str] ,
            metadatas : Optional[Sequence[Dict]] = None 
        ) -> bool : 
        try : 
            self.Collections[collection_name].add(
                ids=[uuid.uuid4() for _ in range(len(docs))],
                documents=docs,
                metadatas=metadatas
            )
            return True
        except : 
            return False 

    @staticmethod
    def GetPdfStr(
            doc : pathlib.Path  # Expecting Pdf 
    ) -> Iterator[str] : 
        if not (doc.suffix == '.pdf') :
            raise ValueError(
                f"Expected .pdf type but got {doc.suffix}"
            )
        for page in PDFPlumberLoader(doc).lazy_load() : 
            yield page.page_content

    @staticmethod
    def GetPdfPageCount(
        doc : pathlib.Path 
    ) -> int : 
        if not (doc.suffix == '.pdf') :
            raise ValueError(
                f"Expected .pdf type but got {doc.suffix}"
            )
        
        for pages in PDFPlumberLoader(doc).lazy_load() : 
            return pages.metadata['total_pages']
        
    
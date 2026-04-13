from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()


from pathlib import Path
import time
import logging

from Agent import Agentutils
from langgraph.graph import StateGraph , END , START

from Agent.AgntResearcher import AddFile , Researcher
from Agent.AgntEditor import Editor
from Agent.AgntWriter import Writter


if not AddFile(path=Path('./data/lORA.pdf')):
    print("False")

def define_workflow() : 
    workflow = StateGraph(Agentutils.AgentState)

    workflow.add_node("researcher", Researcher)
    workflow.add_node("writer", Writter)
    workflow.add_node("grader", Editor)


    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "grader")

    def decide_to_finish(state):
        if state["is_sufficient"]:
            return "end"
        return "researcher"

    workflow.add_conditional_edges(
        "grader",
        decide_to_finish,
        {
            "end": END,
            "researcher": "researcher"
        }
    )

    compiled_workflow = workflow.compile()
    return compiled_workflow



app = FastAPI(title="Pdf Searcher")

compiled_workflow = define_workflow()

@app.get("/lora")
def pdf_search(request : str ):
    start = time.time()
    result = compiled_workflow.invoke(
        {
            'user_input'  : request
        }
    )
    logging.info(f"Time Taken to response :{time.time() - start}s")
    return {'soltuion' : result['solution'] }
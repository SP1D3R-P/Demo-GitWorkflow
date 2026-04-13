
from Agent.Agentutils import AgentState , Summarizer_Model
from langchain_core.prompts import PromptTemplate

from Agent import Agentutils

WriterPrompt  = PromptTemplate.from_template(
    Agentutils.reformat_docstring(
        """
        ### ROLE
        You are a Professional Technical Writer. Your goal is to transform raw research data into a structured, polished report that is easy for stakeholders to read.

        ### OBJECTIVE
        Create a comprehensive response to the USER QUERY using ONLY the provided RESEARCH NOTES. 

        ### INPUT DATA
        - USER QUERY: {query}
        - RESEARCH NOTES: 
        {research_notes}

        ### WRITING GUIDELINES
        1. **Strict Fidelity**: Use only the information provided in the RESEARCH NOTES. If the notes are insufficient, state clearly what is missing rather than making up details.
        2. **Structure**: Use Markdown formatting. Use bold headers, bullet points for lists, and concise paragraphs.
        3. **Tone**: Maintain a professional, objective, and neutral corporate tone.
        4. **No Fluff**: Do not start with "Here is your report" or "Based on the notes." Dive straight into the information.
        5. **Clarity**: Synthesize the information logically. Do not just list the chunks; group related facts together.

        ### REPORT STRUCTURE
        - **Executive Summary**: A 2-sentence overview of the answer.
        - **Detailed Findings**: The core facts organized by sub-topics.
        - **Conclusion/Next Steps**: (Only if supported by the notes).

        ### FINAL CHECK
        Before finishing, ask yourself: "Did I include a single fact, name, or number that was not in the Research Notes?" If yes, remove it.
        """
    )
)


def Writter(State : AgentState):
    
    global WriterPrompt

    prompt = WriterPrompt.format(
        query=State['query'],
        research_notes=State['research_notes']
    )

    request = Summarizer_Model.invoke(prompt)
    return {'solution':request.content}

    

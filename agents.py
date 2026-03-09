# agents.py

from typing_extensions import TypedDict
from typing import List

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from rag.ragapp import rag_query


class State(TypedDict):
    data: str
    parsed_json: dict
    topic: str
    rag_content: str
    rag_sources: List[str]
    solution: str
    verified_solution: str
    solution_clear: bool
    explanation: str
    history: List[str]


llm = ChatGroq(model="openai/gpt-oss-120b")


# PARSER
def parse_data(state: State):

    prompt = f"""
Extract structured information from this math problem.

Problem:
{state["data"]}

Return JSON:

{{
 "problem_text":"",
 "topic":"",
 "variables":[],
 "constraints":[]
}}
"""

    response = llm.invoke(prompt)

    try:
        import json
        parsed = json.loads(response.content)
    except:
        parsed = {"problem_text": state["data"]}

    return {"parsed_json": parsed}


# TOPIC
def detect_topic(state: State):

    prompt = f"""
Identify the math topic.

Problem:
{state["data"]}

Return ONE word:

algebra
probability
calculus
linear_algebra
other
"""

    topic = llm.invoke(prompt).content.lower().strip()

    return {"topic": topic}


# ROUTER
def intent_router(state: State):

    topic = state["topic"]

    if topic in ["algebra", "probability", "calculus", "linear_algebra"]:
        return "rag"

    return "solver"


# RAG
def rag(state: State):

    rag_result = rag_query(state["data"])

    return {
        "rag_content": rag_result.get("context", ""),
        "rag_sources": rag_result.get("sources", [])
    }


# SOLVER
def solver(state: State):

    prompt = f"""
Solve this math problem step by step.

Problem:
{state["data"]}

Helpful formulas:
{state.get("rag_content","")}

Return final answer using LaTeX where needed.
"""

    solution = llm.invoke(prompt)

    return {"solution": solution.content}


# VERIFIER
def verifier(state: State):

    prompt = f"""
Verify the solution.

Problem:
{state["data"]}

Solution:
{state["solution"]}

Check correctness and reasoning.
"""

    verified = llm.invoke(prompt)

    return {"verified_solution": verified.content}


# CLARITY CHECK
def is_solution_clear(state: State):

    prompt = f"""
Check if this solution is clear.

Solution:
{state["verified_solution"]}

Return True or False
"""

    response = llm.invoke(prompt).content.strip()

    clear = "true" in response.lower()

    return {"solution_clear": clear}


# HITL ROUTER
def hitl_router(state: State):

    if state["solution_clear"]:
        return "explainer"

    return "human"


# HUMAN LOOP
def human_loop(state: State):

    human_solution = interrupt(
        {
            "message": "Solution unclear. Please edit the solution.",
            "solution": state["solution"]
        }
    )

    return {"solution": human_solution}


# EXPLAINER
def explainer(state: State):

    prompt = f"""
Explain this solution clearly like a teacher.

Problem:
{state["data"]}

Solution:
{state["solution"]}

Use step-by-step explanation.
"""

    explanation = llm.invoke(prompt)

    return {"explanation": explanation.content}


# MEMORY
def update_memory(state: State):

    history = state.get("history", [])

    history.append(
        f"Problem: {state['data']}\nSolution: {state['solution']}"
    )

    return {"history": history}


# GRAPH
workflow = StateGraph(State)

workflow.add_node("parse_data", parse_data)
workflow.add_node("detect_topic", detect_topic)
workflow.add_node("rag", rag)
workflow.add_node("solver", solver)
workflow.add_node("verifier", verifier)
workflow.add_node("clarity", is_solution_clear)
workflow.add_node("human", human_loop)
workflow.add_node("explainer", explainer)
workflow.add_node("memory", update_memory)

workflow.add_edge(START, "parse_data")
workflow.add_edge("parse_data", "detect_topic")

workflow.add_conditional_edges(
    "detect_topic",
    intent_router,
    {
        "rag": "rag",
        "solver": "solver"
    }
)

workflow.add_edge("rag", "solver")

workflow.add_edge("solver", "verifier")
workflow.add_edge("verifier", "clarity")

workflow.add_conditional_edges(
    "clarity",
    hitl_router,
    {
        "human": "human",
        "explainer": "explainer"
    }
)

workflow.add_edge("human", "explainer")

workflow.add_edge("explainer", "memory")
workflow.add_edge("memory", END)


memory = MemorySaver()

chain = workflow.compile(checkpointer=memory)
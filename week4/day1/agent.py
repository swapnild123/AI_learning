import os
import json
import ast
import operator

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# ============================================================
# PART 1 — LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")


# ============================================================
# PART 2 — CREATE CLIENTS
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

MODEL = "openai/gpt-oss-120b"


# ============================================================
# PART 3 — WEB SEARCH TOOL
# ============================================================

def web_search(query: str) -> str:
    """
    Search the internet using Tavily and return useful
    web search results as a string.
    """

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = response.get("results", [])

        if not results:
            return "No web search results were found."

        formatted_results = []

        for result in results:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")

            formatted_results.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Web search error: {str(e)}"


# ============================================================
# PART 4 — SAFE CALCULATOR TOOL
# ============================================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_math(node):
    """
    Recursively evaluate only allowed mathematical AST nodes.
    """

    if isinstance(node, ast.Expression):
        return _evaluate_math(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator is not allowed.")

        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)

        return ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator is not allowed.")

        operand = _evaluate_math(node.operand)

        return ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Invalid mathematical expression.")


def calculate(expression: str) -> str:
    """
    Calculate a basic mathematical expression.

    Example:
        2 * 2
        (10 + 5) / 3
        2 ** 8
    """

    try:
        tree = ast.parse(expression, mode="eval")

        result = _evaluate_math(tree)

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ============================================================
# PART 5 — DEFINE TOOLS FOR GROQ
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for current, recent, factual, "
                "or up-to-date information. Use this tool when the "
                "user asks about information that may require web search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query to search for on the internet."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Calculate a mathematical expression. "
                "Use this tool for arithmetic and mathematical calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The mathematical expression to calculate, "
                            "for example '2 * 2' or '(10 + 5) / 3'."
                        ),
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]


# ============================================================
# PART 6 — MAP TOOL NAMES TO PYTHON FUNCTIONS
# ============================================================

available_functions = {
    "web_search": web_search,
    "calculate": calculate,
}


# ============================================================
# PART 7 — AI AGENT
# ============================================================

def run_agent(query: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI agent. "
                "You have access to a web search tool and a calculator tool. "
                "Use web_search when current or internet information is needed. "
                "Use calculate when mathematical computation is needed. "
                "Use the information returned by tools to produce an accurate "
                "final answer. Do not invent tool results."
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    # --------------------------------------------------------
    # First LLM call
    # Model decides whether a tool is required
    # --------------------------------------------------------

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    response_message = response.choices[0].message

    # --------------------------------------------------------
    # If no tool is needed, return model's answer
    # --------------------------------------------------------

    if not response_message.tool_calls:
        return response_message.content or ""

    # --------------------------------------------------------
    # Add assistant tool-call message
    # --------------------------------------------------------

    messages.append(response_message)

    # --------------------------------------------------------
    # Execute every requested tool
    # --------------------------------------------------------

    for tool_call in response_message.tool_calls:

        function_name = tool_call.function.name

        try:
            function_args = json.loads(
                tool_call.function.arguments
            )
        except json.JSONDecodeError:
            function_result = "Error: Invalid tool arguments."

        else:
            function_to_call = available_functions.get(
                function_name
            )

            if function_to_call is None:
                function_result = (
                    f"Error: Unknown tool '{function_name}'."
                )

            else:
                try:
                    function_result = function_to_call(
                        **function_args
                    )
                except Exception as e:
                    function_result = (
                        f"Tool execution error: {str(e)}"
                    )

        # Add tool result to conversation
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_result),
            }
        )

    # --------------------------------------------------------
    # Second LLM call
    #
    # Tool results are now available to the model.
    # We force no additional tools because this simple Day 1
    # agent uses one tool-calling round.
    # --------------------------------------------------------

    final_response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="none",
        temperature=0,
    )

    final_answer = final_response.choices[0].message.content

    return final_answer or ""


# ============================================================
# PART 8 — RUN FROM TERMINAL
# ============================================================

if __name__ == "__main__":

    print("AI Agent")
    print("Tools: Web Search + Calculator")
    print("Type 'exit' to stop.\n")

    while True:

        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            answer = run_agent(query)

            print(f"\nAgent: {answer}\n")

        except Exception as e:
            print(f"\nError: {e}\n")
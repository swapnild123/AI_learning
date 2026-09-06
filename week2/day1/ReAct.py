import os
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from groq import Groq 
import re

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kha hai")

client=Groq(api_key=my_api_key)
model="qwen/qwen3.6-27b"

def get_product_price(product):
    if product == 'iphone 17':
        return 1000
    elif product == 'iphone 15':
        return 500
    else:
        return 0

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc_error!"

tools={
    "get_product_price":get_product_price,
    "calculator":calculator
}
system_prompt= """
you are a shopping assistant.

you have these tools:
get_product_price(product)
calculator(expression)
IMPORTANT:
Call tools exactly like these examples:

Action: get_product_price("iphone 17")
Action: calculator("5000-1000")

follow these rules:

1. decide what you need to do next.
2. call only one tool at a time.
3. after writing an action , stop immediately.
4. never guess or invent a tool result.
5. wait until you recieve an observation.
6. then decide your next action.
7. when the task is completed give me final answer.

Format:

Thought: what you need to do
Action: tool_name(argument)

IMPORTANT: Do NOT use native/function/tool calling.
You must write the Thought and Action as plain text.
After Action, stop and wait for Observation.

when finished:

Final answer: your answer
"""
def run_agent(question):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):

        print("\n------------------")
        print("STEP", step + 1)
        print("------------------")

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content

        print(answer)

        # Agent has finished
        if "Final Answer:" in answer:
            break


        # Find the Action
        match = re.search(
        r'Action:\s*(\w+)\("([^"]*)"\)',
        answer
    )


        if match:

            tool_name = match.group(1)

            tool_input = match.group(2)

            tool_input = tool_input.strip()

            tool_input = tool_input.strip('"')


            # Run the tool
            if tool_name in tools:

                tool = tools[tool_name]

                observation = tool(tool_input)

            else:

                observation = "Tool not found"


            print(
                "Observation:",
                observation
            )


            # Add LLM response to memory
            messages.append({
                "role": "assistant",
                "content": answer
            })


            # Give tool result back to LLM
            messages.append({
                "role": "user",
                "content":
                    "Observation: "
                    + str(observation)
            })
            sleep(5)



prompt="""
I have 5000 rupees. What is the price of an iphone 17?
and how much money will I have left?
"""
run_agent(prompt)
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kha hai bhai")

client=Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"


def llm_ans(prompt):
    message={
        "role":"user",
        "content": prompt
    }
    messages=[message]
    response = client.chat.completions.create(
    model=model,
    messages=messages
)
    ans=response.choices[0].message.content
    return ans

bad_prompt="""
#ROLE
You are a support assistant at mobile/laptop company
#TASK 
you have to classify the isuue in a category
#CONSTRAINT
You have to classify isuue in any of three categories namely as Technical, Return, Billing.
#OUTPUT FORMAT
your answer should be in one world only. the word should be from category mentioned in constraint
#FALLBACK
if you get any unrelated isuue then return OTHER
this is a user complaint:
laptop is bad want its refund
"""
print(llm_ans(bad_prompt))
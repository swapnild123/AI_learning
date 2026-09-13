import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api key kaha hai bhai")

client=Groq(api_key=my_api_key)
model="qwen/qwen3.6-27b"

# step1
knowledge_base={
    "bindki":"raghav healthcare centre is top and best hospital in bindki",
    "services": "this provides top level healthcare facilities"
}
#step2 retrieval
def retrieve_info(question):
    question=question.lower()
    if "bindki" in question:
        return knowledge_base["bindki"]
    elif "services" in question:
        return knowledge_base["services"]
    else:
        return None

def ask_llm(question):
    context=retrieve_info(question)
    sys_prompt=f"""answer in one line only. answer only based on this context. context: {context}"""
    system_message={
        "role":"system",
        "content":sys_prompt
    }
    message={
        "role":"user",
        "content":question
    }
    messages=[system_message,message]
    response=client.chat.completions.create(model=model,messages=messages, max_tokens=500)
    answer=response.choices[0].message.content
    return answer

question="do you know raghav healthcare centre bindki"
print(ask_llm(question))


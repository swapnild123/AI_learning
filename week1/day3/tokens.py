import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai") 

client=Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"
# 3prompts
prompt1="hi"
prompt2="time travel in detail"
prompt3="give 1000 words essay on machine learning"

prompts= [prompt1, prompt2, prompt3]
for prompt in prompts:
    message={
    "role":role,
    "content":prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages)
    usage= response.usage
    print(f"prompt: {prompt} -->your tokens: {usage.prompt_tokens} completion_tokens: {usage.completion_tokens} total tokens: {usage.total_tokens}")
# print(response)

# answer=response.choices[0].message.content
# print(answer)

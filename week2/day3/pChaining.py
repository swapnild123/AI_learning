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

JD="""
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""

RESUME="""
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git

Projects:
Built a food delivery backend using
FastAPI and MySQL.

Deployed applications using Docker.
"""

def ask_llm(system_prompt, user_prompt):
    sys_msg={
        "role":"system",
        "content":system_prompt
    }
    user_msg={
        "role":"user",
        "content":user_prompt
    }
    messages=[sys_msg, user_msg]
    response=client.chat.completions.create(model=model, messages=messages, max_tokens=500)
    answer=response.choices[0].message.content
    return answer

def step1_res_extract(RESUME):
    system_prompt="""
    you are a HR assistant. extract all the skills from candidate resume provided.
    only provide skills no other information. do not invent any skill by yourself.
    """
    user_prompt=f"""
    extract the skills from this resume
    {RESUME}
    """
    return ask_llm(system_prompt, user_prompt)

def step1_JD_extract(JD):
    system_prompt="""
    you are a HR assistant. extract all the skills from candidate JD provided.
    only provide skills no other information. do not invent any skill by yourself.
    """
    user_prompt=f"""
    extract the skills from this JD
    {JD}
    """
    return ask_llm(system_prompt, user_prompt)

def step3_match(candidate,jd):
    system_prompt="""
    you are a HR assistant. compare the skills of candidate from his resume with the skills in job description and provide a scrore
    from 1 to 100. also provide a final verdict whether this candidate is good fit for role or not.
    """
    user_prompt=f"""
    compare and match the skills
    JD:
    {jd}
    Candidate:
    {candidate}
    """
    return ask_llm(system_prompt, user_prompt)

candidate=step1_res_extract(RESUME)
sleep(2)
jd=step1_JD_extract(JD)
sleep(2)
score=step3_match(candidate,jd)
print (score)

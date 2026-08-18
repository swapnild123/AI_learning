import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai") 

client=Groq(api_key=my_api_key)

model = "openai/gpt-oss-20b"
role="user"

from pydantic import BaseModel
class ticket(BaseModel):
    name:str
    email:str
    issue:str
    phone:int

schema=ticket.model_json_schema()

response_format={
    "type":"json_object"
}

system_prompt=f"""
extract the personal information from the ticket strictly based on this schema and give me json output.
{schema}
"""

message_system={
    "role":"system",
    "content":system_prompt
}

text="hello my name is swapnil dwivedi I purchased a iphone but now its generating lot of problems like charging issue. my phone number is 91440088 and email is abc@gmail.com"

prompt=f"""
this is a customer ticket. please extract personal information from this.
{text}
"""
message={
    "role":role,
    "content":prompt
}

messages=[message_system, message]

response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
##print(response)

answer=response.choices[0].message.content
print(answer)

import json
raw_json=answer
data_file=json.loads(raw_json)
ticket=ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)
print(ticket.phone)

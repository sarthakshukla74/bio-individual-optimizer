import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel,Field
from typing import Optional
from enum import Enum
import instructor
from groq import Groq
import base64

client=instructor.from_groq(Groq(api_key=os.getenv("GROQ_API_KEY")),mode=instructor.Mode.JSON)

class Feeling(str,Enum):
    easy="easy"
    moderate="moderate"
    hard="hard"
    maximal="maximal"
class ExerciseSet(BaseModel):
    exercise: str = Field(description="Name of exercise for eg. bench press")
    sets: Optional[int]=Field(default=None,description="Number of sets performed of an exercise")
    reps: Optional[int]= Field( default=None,description="Number of reps")
    weight_kg: Optional[float]=Field(default=None,description="Weight lifted in kg")
    is_pr: bool=Field(default=False,description="True if personal record")
    feeling: Optional[Feeling]=Field(default=None,description="Subjective Difficulty")

class WorkoutLog(BaseModel):
    exercises:list[ExerciseSet]
    sleep_hours:Optional[float]=Field(default=None,description="Hours slept last night. Put this here, not inside exercises.")
    notes:Optional[str]=Field(default=None,description="Any extra context")


def extract(raw_note: str)->WorkoutLog:
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=WorkoutLog,
        messages=[
            {
            "role":"system",
            "content": (
     "You are a fitness log parser. Extract structured workout data from messy gym notes. "
    "RULES: "
    "Any subjective comments like 'felt heavy', 'slept like a baby' should go in the notes field. "
    "'X for Y'or 'X for Y reps' means weight=X, reps=Y, sets=1 unless sets are explicitly mentioned and 'X sets of any exercise' means number of sets=X. "
    "All weights must be stored in kg as a number. If the note uses lbs, do the conversion yourself and store only the final kg value. "
    "sleep_hours is a top level field, never put it inside exercises. "
    "If something is unclear, omit it — never guess. "
    "RPE is a scale of 1 to 10 which goes higher as difficulty increases so use it to determine feeling. "
    "If the number of sets is explicitly not mentioned for eg text says did bicep curls with 20kg got 5 reps it means the number of sets is one. "
    "EXAMPLE: '95kg for 3 reps' → sets=1, reps=3, weight_kg=95.0 "
"EXAMPLE: '4 sets of lat pulldowns at 60kg' → sets=4, reps=null, weight_kg=60.0 "
"EXAMPLE: 'OHP 3x5 at 60kg' → sets=3, reps=5, weight_kg=60.0 "
"EXAMPLE: 'Got X reps'-> sets=1,reps=3. "
"EXAMPLE: 'felt heavy as hell' → feeling=hard "
"EXAMPLE: 'RPE 8 overall' → apply feeling=hard to all exercises in the session "
            )
            },
            {
                "role":"user",
                "content":raw_note
            }
            
            ]

    )


def image_to_base64(image_path:str)->str:
    with open(image_path,"rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


from openai import OpenAI

openai_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_raw=Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_from_image(image_path: str)->WorkoutLog:
    image_data=image_to_base64(image_path)

    response=groq_raw.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role":"user",
                "content":[
                    {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/jpeg;base64,{image_data}"
                    }
                
            },
            {
                "type":"text",
                "text": "Transcribe every word you see in this image as plain text exactly as written. Include all numbers, units, and descriptive words like 'felt good' or 'sleep 7 hours'. Just the text, nothing else."
            }
        ]
        }
        ]
    )

    raw_text=response.choices[0].message.content
    return extract(raw_text)


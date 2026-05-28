import os
from urllib import response
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from extractor import WorkoutLog, ExerciseSet
from extractor import extract
import io
import sys
import csv
import datetime
import pandas as pd
import numpy as np
groq_raw= Groq(api_key=(os.getenv("GROQ_API_KEY")))

def cleaner_agent(log: WorkoutLog)->WorkoutLog:
 for ex in log.exercises:

    response= groq_raw.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role":"system",
                "content":(
                    "You are an exercise name normaliser. Return only the standard name for the exercise given. Examples: 'Squatted' -> 'squat', 'Benched' -> 'bench press', 'OHP' -> 'overhead press'. Return only the name, nothing else."
                )
            },
            {
            "role":"user",
            "content": ex.exercise
            }
        ]
    )
    ex.exercise=response.choices[0].message.content.strip()
 return log


def historian_agent(log: WorkoutLog)->str:
    today=datetime.date.today().isoformat()
    with open("master_log.csv","a",newline="") as f:
      writer=csv.writer(f)
      for ex in log.exercises:
        writer.writerow([today, ex.exercise, ex.sets, ex.reps, ex.weight_kg, ex.is_pr, ex.feeling, log.sleep_hours, log.notes])
    
    return "master_log.csv"

if __name__=="__main__":
    log=extract("Squatted 100 for 5, Benched 80kg for 3, OHP 3x5 at 60kg,slept 7 hours,felt great")
    cleaned=cleaner_agent(log)
    path=historian_agent(cleaned)
    print(f"Logged workout to {path}")

def analyst_agent(question: str)->str:
   df=pd.read_csv("master_log.csv", names=["date","exercise","sets","reps","weight_kg","is_pr","feeling","sleep_hours","notes"])
   df['sleep_hours'] = pd.to_numeric(df['sleep_hours'], errors='coerce')
   df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
   df['reps'] = pd.to_numeric(df['reps'], errors='coerce')
   df['sets'] = pd.to_numeric(df['sets'], errors='coerce')
   df['is_pr'] = df['is_pr'].map({'True': True, 'False': False, True: True, False: False})
   response=groq_raw.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
       {
          "role":"system",
            "content":(
                 "You are a data analyst. You have access to a pandas dataframe called df with columns: "
                    "date, exercise, sets, reps, weight_kg, is_pr, feeling, sleep_hours, notes. "
                    "Write only Python pandas code to answer the question. No explanation, no markdown, just raw executable Python code. "
                    "Use print() to output your answer."
                    "If a query returns empty results, print a helpful message explaining why rather than throwing an error. "
"Always use try/except around operations that might fail on empty data. "
"The is_pr column contains True/False boolean values. "
"Always check if filtered data is empty before calling operations like idxmax or argmax. "
            )
       },
       {
            "role":"user",
                "content": question
       }
    ]
     
   )

   code=response.choices[0].message.content.strip()
   code = code.replace("```python", "").replace("```", "").strip()
   print(f"CODE: {code}")
   buffer = io.StringIO()
   sys.stdout = buffer
   try:
    exec(code, {"df": df, "pd": pd})
   finally:
    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
   if not output:
    return "The query returned no results. Your data may not contain the information needed to answer this question."
   return output

def critic_agent(log: WorkoutLog)->dict:
   response=groq_raw.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
         {
            "role":"system",
            "content":(
                  "You are a workout critic. You take in a workout log and give feedback on how to improve. Focus on one area of improvement, such as exercise selection, volume, intensity, recovery, or consistency. Be constructive and specific with your feedback."
            )
         },
         {
            "role":"user",
        "content": (f"Here is the workout log: {log}. What is one specific area of improvement and how can it be improved?"
        )
         }
        ]

   )
   feedback=response.choices[0].message.content.strip()
   return {"feedback": feedback}
    
def checker_agent(log: WorkoutLog)->dict:
   issues=[]

   for ex in log.exercises:
      if ex.weight_kg and ex.weight_kg>300:
            issues.append(f"Suspicious weight for {ex.exercise}: {ex.weight_kg} kg")
      if ex.reps and ex.reps>50:
            issues.append(f"Suspicious reps for {ex.exercise}: {ex.reps}")
      if ex.sets and ex.sets>6:
            issues.append(f"Suspicious sets for {ex.exercise}: {ex.sets}")
      if not ex.exercise or len(ex.exercise)<2:
            issues.append("Exercise name is too short or missing")
      if log.sleep_hours and log.sleep_hours>24:
            issues.append(f"Impossiblers: {log.sleep_hours}")

   return{
        "is_valid": len(issues)==0,
        "issues": issues
    }
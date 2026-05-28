import streamlit as st
import plotly.express as px
import pandas as pd
from extractor import extract_from_image,extract
from agents import cleaner_agent,historian_agent,analyst_agent,critic_agent,checker_agent
from rag import rag_answer

st.title("Bio-Individual-Optimiser")
st.caption("Your personal AI fitness analyst")

st.subheader("Log your workout")

tab1,tab2=st.tabs(["Text","Image"])

with tab1:
    note=st.text_area("Enter your workout details",placeholder="E.g. Squatted 100 for 5, Benched 80kg for 3, OHP 3x5 at 60kg,slept 7 hours,felt great")
    if st.button("Extract and Save"):
        with st.spinner("Analyzing..."):
            log=extract(note)
            cleaned=cleaner_agent(log)
            path=historian_agent(cleaned)
            st.success("Workout Logged")
            st.json(cleaned.model_dump())

with tab2:
    photo=st.file_uploader("Upload a photo of your workout note",type=["jpg","jpeg","png"])
    if st.button("Extract from Photo"):
        if photo:
            with open("temp_image.jpg","wb") as f:
                f.write(photo.getbuffer())
            with st.spinner("Reading image.."):
                    log=extract_from_image("temp_image.jpg")
                    cleaned=cleaner_agent(log)
                    path=historian_agent(cleaned)
                    st.success("Workout Logged")
                    st.json(cleaned.model_dump())


st.subheader("Your performance over time")

if st.button("Load Chart"):
    st.session_state['show_chart'] = True

if st.session_state.get('show_chart'):
     try:
          df=pd.read_csv("master_log.csv", names=["date","exercise","sets","reps","weight_kg","is_pr","feeling","sleep_hours","notes"])
          df['weight_kg']=pd.to_numeric(df['weight_kg'],errors='coerce')

          exercise_list=df['exercise'].unique().tolist()
          selected=st.selectbox("Select Exercise",exercise_list)

          filtered=df[df['exercise']==selected]
          
          if len(filtered) < 5:
              fig = px.scatter(filtered, x='date', y='weight_kg', title=f"{selected} — weight over time")
              st.caption("Not enough data for a line chart — showing scatter plot. Log more workouts to see trends.")
          else:
             fig = px.line(filtered, x='date', y='weight_kg', title=f"{selected} — weight over time")
         
          st.plotly_chart(fig)
     except Exception as e:
          st.error(f"Error: {e}")
     
    
st.subheader("Ask about your data")

question=st.text_input("Ask a question about your workouts",placeholder="E.g. Which exercise do I have the most PRs in? How does my sleep affect my performance? What was my best deadlift?")
if st.button("Analyse"):
    if not question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            result = analyst_agent(question)
            st.write(result)
st.subheader("Ask the science")
science_question=st.text_input("Ask a science question",placeholder="E.g. Why does sleep deprivation reduce strength? How long should I rest between sets? What are the best exercises for building a bigger chest?")
if st.button("Search papers"):
    with st.spinner("Searching research papers..."):
        result=rag_answer(science_question)
        st.write(result)

if st.button("Get Feedback on Last Workout"):
    with st.spinner("Analyzing..."):
        df = pd.read_csv("master_log.csv", names=["date","exercise","sets","reps","weight_kg","is_pr","feeling","sleep_hours","notes"])
        last_row = df.iloc[-1]
        from extractor import ExerciseSet, WorkoutLog
        log = WorkoutLog(
        exercises=[ExerciseSet(
        exercise=last_row['exercise'],
        sets=last_row['sets'],
        reps=last_row['reps'],
        weight_kg=last_row['weight_kg'],
        is_pr=last_row['is_pr'],
        feeling=last_row['feeling'] if pd.notna(last_row['feeling']) else None
    )],
       sleep_hours=last_row['sleep_hours'] if pd.notna(last_row['sleep_hours']) else None,
       notes=last_row['notes'] if pd.notna(last_row['notes']) else None
    )
    feedback=critic_agent(log)
    st.markdown(feedback['feedback'])
    check=checker_agent(log)
    if not check['is_valid']:
      st.warning("Data quality issue found")
      for issue in check['issues']:
        st.write(f"- {issue}")
    else:    st.success("No data quality issues found")
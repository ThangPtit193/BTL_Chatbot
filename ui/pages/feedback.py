import streamlit as st
import pandas as pd
from datetime import datetime

# Feedback page
# input text for feedback
# save feedback to file
def feedback():
    st.title("📝 Feedback")
    st.write("Please give us your feedback")
    # save feedback and time to csv file
    feedback = st.text_area("Feedback", height=200)
    if st.button("Submit"):
        feedback_df = pd.DataFrame({"feedback": [feedback], "time": [datetime.now()]})
        feedback_df.to_csv("ui/logs/feedback.csv", mode="a", index=False, header=False)
        st.success("Feedback saved")

feedback()
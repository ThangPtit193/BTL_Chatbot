import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Welcome to saturn!")

st.write("# Saturn Demo - Explore the world")
st.markdown("""
This demo takes its data from a selection of VA Projects including Fschool, Ftech, Timi Idol

Ask any question on this topic and see if saturn can find the correct answer to your query!
""")
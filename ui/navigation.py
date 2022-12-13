from PIL import Image
import streamlit as st

image = Image.open("images/saturn.png")

st.image(image)
# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )

st.sidebar.success("Welcome to saturn!")

st.write("<h1 style='text-align: center; color: grey;'>Saturn Demo - Explore the world</h1>", unsafe_allow_html=True)

st.markdown('''<p style='text-align: center; color: black; font-size:18px;'> 
                This demo takes its data from a selection of VA Projects including Fschool, Ftech, Timi Idol <br>
                Ask any question on this topic and see if saturn can find the correct answer to your query!</p>''',
            unsafe_allow_html=True)

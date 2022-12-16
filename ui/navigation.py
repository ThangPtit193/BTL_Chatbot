from PIL import Image
import streamlit as st
image = Image.open("images/saturn.png")

st.set_page_config(
    page_title="Knowledge Retrieval",
    page_icon=image,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://trello.com/b/8zAozWIB/knowledge-retriever',
        'Report a bug': 'https://ftech.ai/',
        'About': 'The services is applied for knowledge retrieval and infer models quickly developed by VA TEAM',
    },
)
# center image
col_1, col_2, col_3 = st.columns([1, 1, 1])
with col_2:
    st.image(image, width=500)
st.sidebar.success("Welcome to saturn!")

st.write("<h1 style='text-align: center; color: grey;'>Saturn Demo - Explore the world</h1>", unsafe_allow_html=True)

st.markdown('''<p style='text-align: center; color: black; font-size:18px;'> 
                This demo takes its data from a selection of VA Projects including Fschool, Ftech, Timi Idol <br>
                Ask any question on this topic and see if saturn can find the correct answer to your query!</p>''',
            unsafe_allow_html=True)

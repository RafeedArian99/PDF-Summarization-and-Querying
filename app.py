import streamlit as st
import pandas as pd
from io import StringIO

PAGE_CONFIG = {"page_title":"Summarizing and Querying","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


def main():
    page = st.sidebar.radio("Menu", ["Home", "Summarizing", "Querying"])

    if page == "Home":
        show_home()
    elif page == "Summarizing":
        show_summarizing()
    elif page == "Querying":
        show_querying()


def show_home():
    st.title("Summarizing and Querying PDF")
    st.subheader("Choose menu on the left sidebar")
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.7,6,1])

    with col1:
        st.write("")

    with col2:
        st.image('Image.jpg', width=300)

    with col3:
        st.write("")

    st.markdown("<h4 style='text-align: center;'>Hello, how can I help you?</h4>", unsafe_allow_html=True)


def show_summarizing():
    st.title("Summarizing")
    st.write("Upload your PDF file and get a summary")
    uploaded_file = st.file_uploader("Choose a file for summarizing", type=['pdf'], key='summarizing_uploader')
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        # PDF 요약 처리 코드 추가 부분

    col1, col2, col3 = st.columns([2.5, 6, 1])

    with col1:
        st.write("")

    with col2:
        submit_button = st.button('Submit for Summarizing')

    with col3:
        st.write("")
    if submit_button:
        st.write("PDF is being summarized...")
        # PDF 요약 로직 처리


def show_querying():
    st.title("Querying")
    st.write("Upload your PDF file and submit queries")
    uploaded_file = st.file_uploader("Choose a file for querying", type=['pdf'], key='querying_uploader')
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        # PDF 쿼리 처리 준비

    prompt = st.chat_input("Input Query")
    if prompt:
        st.write(f"You: {prompt}")
    # if submit_button:
    #     st.write("Query submitted:", query)
        # PDF에서 쿼리 처리 로직 실행


if __name__ == '__main__':
    main()
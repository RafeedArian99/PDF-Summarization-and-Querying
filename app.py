import streamlit as st
import pandas as pd
from io import StringIO
from queryer import Queryer

PAGE_CONFIG = {
    "page_title": "Summarizing and Querying",
    "page_icon": ":smiley:",
    "layout": "centered",
}
st.set_page_config(**PAGE_CONFIG)


def main():
    page = st.sidebar.radio("Menu", ["Home", "Summarizing", "Querying"])
    if "queryer" not in st.session_state:
        st.session_state.queryer = Queryer()
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

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
    col1, col2, col3 = st.columns([2.7, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.image("Image.jpg", width=300)

    with col3:
        st.write("")

    st.markdown(
        "<h4 style='text-align: center;'>Hello, how can I help you?</h4>", unsafe_allow_html=True
    )


def show_summarizing():
    st.title("Summarizing")
    st.write("Upload your PDF file and get a summary")
    uploaded_file = st.file_uploader(
        "Choose a file for summarizing", type=["pdf"], key="summarizing_uploader"
    )
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        # PDF 요약 처리 코드 추가 부분

    col1, col2, col3 = st.columns([2.5, 6, 1])

    with col1:
        st.write("")

    with col2:
        submit_button = st.button("Submit for Summarizing")

    with col3:
        st.write("")
    if submit_button:
        st.write("PDF is being summarized...")
        # PDF 요약 로직 처리


def show_querying():
    st.title("Querying")
    # st.write("Upload your PDF file and submit queries")
    uploaded_file = st.file_uploader("Choose a file for querying", type=["pdf"])
    if uploaded_file and not st.session_state.file_processed:
        st.write("Processing File...")
        st.session_state.queryer.process(uploaded_file.read())
        st.write("Complete")
        st.session_state.file_processed = True
    elif not uploaded_file and st.session_state.file_processed:
        st.write("File removed")
        st.session_state.file_processed = False

    prompt = st.text_input("Input Query", key="query_input")
    submit_query = st.button("Submit Query")

    if submit_query and prompt and st.session_state.file_processed:
        response = st.session_state.queryer.ask(prompt)
        st.write(f"Response: {response}")


if __name__ == "__main__":
    main()

# main.py
import streamlit as st
import dashboard_tweets, dashboard_iNaturalist

PAGES = {
    "Tweets": dashboard_tweets,
    "iNaturalist": dashboard_iNaturalist
}

def main():
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    PAGES[choice].app()

if __name__ == "__main__":
    main()

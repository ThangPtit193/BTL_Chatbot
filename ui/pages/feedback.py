import google_auth_httplib2
import httplib2
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import HttpRequest
from datetime import datetime
from utils.io import read_feedback_config

@st.experimental_memo
def read_config(file_path):
    return read_feedback_config(file_path)

fback_config = read_config("ui/feedback_config/Feedback_config.yaml")


@st.experimental_singleton()
def connect_to_gsheet():
    # Create a connection object.

    credentials = service_account.Credentials.from_service_account_file(
        fback_config['SERVICE_ACCOUNT_FILE'],
        scopes=[fback_config['SCOPE']],
    )

    # Create a new Http() object for every request
    def build_request(http, *args, **kwargs):
        new_http = google_auth_httplib2.AuthorizedHttp(
            credentials, http=httplib2.Http()
        )
        return HttpRequest(new_http, *args, **kwargs)

    authorized_http = google_auth_httplib2.AuthorizedHttp(
        credentials, http=httplib2.Http()
    )
    service = build(
        "sheets",
        "v4",
        requestBuilder=build_request,
        http=authorized_http,
    )
    gsheet_connector = service.spreadsheets()
    return gsheet_connector

def get_data(gsheet_connector) -> pd.DataFrame:
    values = (
        gsheet_connector.values()
        .get(
            spreadsheetId=fback_config['SPREADSHEET_ID'],
            range=f"{fback_config['SHEET_NAME']}!A:E",
        )
        .execute()
    )

    df = pd.DataFrame(values["values"])
    df.columns = df.iloc[0]
    df = df[1:]
    return df


def add_row_to_gsheet(gsheet_connector, row) -> None:
    gsheet_connector.values().append(
        spreadsheetId=fback_config['SPREADSHEET_ID'],
        range=f"{fback_config['SHEET_NAME']}!A:E",
        body=dict(values=row),
        valueInputOption="USER_ENTERED",
    ).execute()
st.title("📝 Feedback")
st.write("Please give us your feedback")

gsheet_connector = connect_to_gsheet()

# save feedback and time to csv file


with st.form("feedback_form"):
    author = st.text_input("Author")
    feed_backs = st.text_area("Feedback", height=150)
    recommend = st.text_area("Recommend", height=150)
    submit_button = st.form_submit_button("Submit")
    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


if submit_button:
    # check if the feedback is empty
    if feed_backs == "":
        st.error("Please enter your feedback")
        st.stop()
    add_row_to_gsheet(gsheet_connector, [[author, feed_backs, recommend, date_time]])
    st.success("Feedbacks saved")



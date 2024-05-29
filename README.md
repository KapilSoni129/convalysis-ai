# convalysis-ai

## Description

convalysis-ai is an AI powered tool which can be used to analyze conversations between people. It supports any type of transcriptable conversation. Currently, convalysis can be used to summarize conversations, analyse speaker emotions and find out agreement / disagreement between speakers in a conversation. It has been implemented in the form of a backend API, which takes in the transcription of a conversation as input.

## Features

A user can:

1. Get the summary of a conversation.
2. Get the primary and secondary emotions of all speakers present in the conversation.
3. Get agreement scores of a speaker wrt. all the other speakers in the conversation.

For more information regarding the APIs, please refer the API documentation.
For more inforamtion regarding what agreement score is and how it is calculated, please refer to the report.

## Tech Stack

- FastAPI HTTP Server
- PyTorch for training NLP models

## Setup Locally

Clone the repository and run the following commands in the root directory.

```shell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
fastapi run ./app/main.py
```

## Further Information

convalysis-ai was created as an implementation of the work done in the Minor Project in my 6th semester. Please refer to the project report for further information regarding the dataset used to train the model and the algorithm used to calculate agreement scores.

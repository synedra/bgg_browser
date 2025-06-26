
import os
import pandas as pd
import kagglehub
import chromadb
from astrapy import DataAPIClient

from google import genai
from google.genai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from dotenv import load_dotenv
import json
import os
from openai import OpenAI
import cassio
from langchain_community.vectorstores import Cassandra
from langchain.schema import Document

load_dotenv()

# Set your OpenAI API key
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")


ASTRA_TOKEN = os.environ.get("ASTRA_TOKEN")
astraclient = DataAPIClient(token=ASTRA_TOKEN)
db = astraclient.get_database_by_api_endpoint(os.environ.get("ASTRA_ENDPOINT")
)
print(f"Connected to Astra DB: {db.list_collection_names()}")

from IPython.display import Markdown

# set up client
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
# retry helper
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

path = kagglehub.dataset_download("threnjen/board-games-database-from-boardgamegeek")
games_df = pd.read_csv(os.path.join(path, "games.csv"))
games_df = games_df.sort_values(by=['BayesAvgRating'], ascending=False)
print(games_df.columns)

game_collection = db.create_collection("bgg_collection")

records = games_df.to_dict(orient='records')
for record in records:
    

# Generate embeddings
    embedding = OpenAI.Embedding.create(
        input=record,
        model="text-embedding-3-large"  # Or try "text-embedding-3-large" for more precision
    )
    print ("Inserting record:", record['Name'])
    try:
        game_collection.insert_one(record,vector=embedding['data'][0]['embedding'])
        print(f"Inserted record {record['Name']} successfully.")
    except Exception as e:
        print(f"Error inserting record {record['Name']}: {e}")
        print (record)
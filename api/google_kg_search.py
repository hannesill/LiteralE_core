import os
from dotenv import load_dotenv
import requests
import json

load_dotenv(dotenv_path="../.env")

KG_SEARCH_API_KEY = os.getenv("GOOGLE_KG_SEARCH_API_KEY")
print(KG_SEARCH_API_KEY)


# Convert freebase ids to names
def get_entity_name(entity_id):
    url = f"https://kgsearch.googleapis.com/v1/entities:search?ids={entity_id}&key={KG_SEARCH_API_KEY}"
    response = requests.get(url)
    try:
        entity_name = json.loads(response.text)["itemListElement"][0]["result"]["name"]
    except:
        print(f"Error for entity with id {entity_id}: {response.text}")
        entity_name = entity_id
    return entity_name

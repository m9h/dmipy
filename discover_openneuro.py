
import requests
import json
import pandas as pd

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

QUERY = """
query {
  dataset(id: "ds006557") {
    id
    public
    latestSnapshot {
      tag
      summary {
        modalities
        subjects
        sessions
      }
    }
    metadata {
      datasetName
    }
  }
}
"""

def search_openneuro():
    print("Querying OpenNeuro GraphQL API...")
    try:
        response = requests.post(GRAPHQL_URL, json={'query': QUERY}, headers={'Content-Type': 'application/json'})
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return

        data = response.json()
        print("Response:", json.dumps(data, indent=2))
        
        ds = data.get('data', {}).get('dataset', {})
        if not ds:
            print("Dataset ds006557 not found or not accessible (maybe private?).")
            return
            
        print(f"Dataset: {ds.get('id')}")
        print(f"Public: {ds.get('public')}")
        snap = ds.get('latestSnapshot', {})
        if snap:
             print(f"Latest Snapshot: {snap.get('tag')}")
             print(f"Modalities: {snap.get('summary', {}).get('modalities')}")
        else:
             print("No snapshots found.")

            
    except Exception as e:
        print(f"Failed to query OpenNeuro: {e}")

if __name__ == "__main__":
    search_openneuro()

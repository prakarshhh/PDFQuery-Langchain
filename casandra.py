from cassandra.cluster import Cluster
import os
try:
    cluster = Cluster()
    session = cluster.connect()
    print("Cassandra driver is working correctly.")
except Exception as e:
    print(f"Error: {e}")
    
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:FKFxqdtvqpUXcAtIYLsXEJOM:2b4ce8649b7ff79151384b8e11f9d69aa3c421ee26e8369c66d3bb67bce474fc" # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "b4c91d3a-5f95-40cb-a174-149faf745615" # enter your Database ID
OPENAI_API_KEY = "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" # enter your OpenAI key

# Print the values (just to check)
print("Astra DB Token:", ASTRA_DB_APPLICATION_TOKEN)
print("Astra DB ID:", ASTRA_DB_ID)
print("OpenAI API Key:", OPENAI_API_KEY)
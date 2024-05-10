""""
`utils.list_available_collections` module: Lists all Chroma DB collections stored in a local SQLite database.
"""
import os
import sqlite3

def list_available_collections() -> list:
    """
    Returns a list of all the Chroma DB collections stored in the local SQLite database,
    based on the 'collection_metadata' table or a similar mechanism.
    
    The path to the database is determined by the VECTOR_SEARCH_PATH environment variable.
    
    Returns:
    - A list of collection names.
    """
    collections = []
    
    vector_search_path = os.environ.get("VECTOR_SEARCH_PATH")
    db_path = os.path.join(vector_search_path, "chroma.sqlite3")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Assuming 'collection_metadata' has a column 'name' that lists collection names
        cursor.execute("SELECT DISTINCT name FROM collections;")
        collection_names = cursor.fetchall()
        
        for name in collection_names:
            collections.append(name[0])
        
        conn.close()
        
    except Exception as e:
        print(f"Error accessing the Chroma DB: {e}")
    
    return collections


from opensearchpy import OpenSearch, RequestsHttpConnection
import os

def connect_to_source(query, index):
    os_client = OpenSearch(
        hosts=[os.environ['OPENSEARCH_ENDPOINT']],
        http_auth=(os.environ['OPENSEARCH_USERNAME'], os.environ['OPENSEARCH_PASSWORD']),
        connection_class=RequestsHttpConnection
    )

    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["*"]
            }
        },
        "size": 10,
        "_source": ["id", "title", "url", "plain_text", "month", "day", "year", "source_file"]
    }

    res = os_client.search(index=index, body=body)

    results = []
    for hit in res['hits']['hits']:
        result = {
            "title": hit["_source"].get("title"),
            "url": hit["_source"].get("url"),
            "snippet": hit["_source"].get("plain_text")[:400],
            "id": hit["_source"].get("id"),
            "month": hit["_source"].get("month"),
            "day": hit["_source"].get("day"),
            "year": hit["_source"].get("year"),
            "source_file": hit["_source"].get("source_file"),

        }
        results.append(result)

    return results
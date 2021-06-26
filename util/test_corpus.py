from elasticsearch import Elasticsearch

es = Elasticsearch()

request_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },

    "mappings": {
        "properties": {
            "name": {"type": "keyword"},
            "text": {
                "type": "text",
                "fields": {
                    "english": {
                        "type": "text",
                        "analyzer": "english"
                    }
                }
            },
            "embedding": {"type": "dense_vector", "dims": 768}
        },
        "dynamic_templates": [
            {
                "strings": {
                    "path_match": "*",
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"}}}
        ],
    }
}
# # es.indices.delete(index="pubmed_detailed")
# es.indices.create(index="pubmed_detailed", body=request_body)

print(es.indices.get(index="_all"))
print(es.cat.indices())
print(es.cluster.health(index="pubmed_detailed"))

# res = es.search(index="pubmed_detailed", body={"query": {"match_all": {}}})
# print("Got %d Hits:" % res['hits']['total']['value'])
# for hit in res['hits']['hits']:
#     print("%(text)s" % hit["_source"])
#     exit()

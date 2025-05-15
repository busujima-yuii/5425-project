from whoosh.qparser import QueryParser

from indexer import ix

# 关键词搜索
def search_segments(keyword, limit=10):
    parser = QueryParser("content", schema=ix.schema)
    q = parser.parse(keyword)
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=limit)
        return [
            {
                "start": float(r["start"]),
                "end":   float(r["end"]),
                "text":  r["content"]
            }
            for r in results
        ]

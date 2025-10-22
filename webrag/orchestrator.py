class WebRAG:
    def __init__(self, sources=None, out_dir="exports/"):
        self.sources = sources or []
        self.out_dir = out_dir
        self.docs = []

    def build(self):
        # For now just simulate fetching and chunking
        for src in self.sources:
            page = {
                "url": src["url"],
                "chunks": [{"id": "1", "content": "Example content from " + src["url"]}]
            }
            self.docs.append(page)
        print("Build complete.")

    def export(self):
        # Return the documents
        return self.docs

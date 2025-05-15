from whoosh import index
from whoosh.fields import Schema, TEXT, NUMERIC, ID
import os

# init Whoosh indices
schema = Schema(
    id=ID(stored=True, unique=True),
    content=TEXT(stored=True),
    start=NUMERIC(stored=True, decimal_places=3),
    end=NUMERIC(stored=True, decimal_places=3),
)

indexdir = "indexdir"
if not os.path.exists(indexdir):
    os.mkdir(indexdir)
    ix = index.create_in(indexdir, schema)
else:
    ix = index.open_dir(indexdir)


def build_index(segments):
    """
    segments: [{"start": float, "end": float, "text": str}, …]
    """
    writer = ix.writer()
    for seg in segments:
        writer.add_document(
            id=f"{seg['start']:.3f}_{seg['end']:.3f}",
            content=seg["text"],
            start=seg["start"],
            end=seg["end"]
        )
    writer.commit()


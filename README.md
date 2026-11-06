# Biomaterials Atlas

Embeddings, bibliographic metadata and fixed topic maps for a PubMed-indexed biomaterials collection.

[Explore the atlas](https://www.bastienaymon.com/atlas/) · [Download the data release](https://github.com/bastenefg/biomaterials-atlas/releases/tag/data-2026-11-06)

The release contains **150,060 paper records**, including **146,319 papers with 768-dimensional SPECTER2 embeddings** and **3,741 unmapped records**. It covers records dated **January 1, 2000–October 22, 2026**. The topic hierarchy contains **21 broad themes and 133 fine topics**.

Use the data to retrieve related papers, inspect topic membership, build your own visualizations, or compare methods on the same frozen collection. The download contains the existing vectors and coordinates; no GPU, model download, API key or embedding computation is required.

## Get started

Use Python 3.10 or later. The loader requires NumPy; the downloader uses only Python's standard library.

```bash
git clone https://github.com/bastenefg/biomaterials-atlas.git
cd biomaterials-atlas
python -m pip install -r requirements.txt
python download_data.py
python atlas_data.py 33169642 --k 10
```

The downloader checks the release archive's SHA-256, extracts it into `data/`, verifies every payload file, and removes the temporary archive. It refuses to overwrite an existing data directory. To use another destination:

```bash
python download_data.py --output /path/to/atlas
python atlas_data.py 33169642 --data /path/to/atlas/data
```

You can also download the ZIP and `SHA256SUMS` from the release page and extract the ZIP yourself. The download is **488 MiB** and the extracted data is **698 MiB**. Allow about **2 GB of free disk space** for downloading and extraction. To check extracted files later:

```bash
python download_data.py --verify-only data
```

## Work with the embeddings

```python
from atlas_data import Atlas

atlas = Atlas("data")
print(atlas.embeddings.shape)  # (146319, 768), memory-mapped float32

row = atlas.row_by_pmid["33169642"]
vector = atlas.embeddings[row]

for paper in atlas.related("33169642", k=10):
    print(paper["cosine_similarity"], paper["pmid"], paper["title"])

# More than 25 neighbors triggers an exact cosine scan of the stored vectors.
neighbors = atlas.related("33169642", k=50)

# Coordinates match the rounded values used on the website.
xy = atlas.coordinates(layout="spacious", dimensions=2)
xyz = atlas.coordinates(layout="spacious", dimensions=3)
topics = atlas.topic_assignments()
print(topics[row]["broad_topic"], topics[row]["fine_topic"])
```

Vectors are L2-normalized, so their dot product is cosine similarity. `related()` uses the stored exact top-25 original-space neighbors by default; `exact=True` recomputes the ranking. Rankings exclude the query paper. An unmapped or absent PMID raises a clear error.

PMID is the primary identifier. DOI values are not guaranteed unique or complete. **Embedding row indices always refer to `embedding_rows.csv`, never to a separately sorted metadata table.** All coordinate arrays, topic assignments and neighbor indices share this order. The first 146,319 lines in `papers.jsonl` match that order; the remaining records have `embedding_row: null` and no fabricated vectors or coordinates. Keep PMID and row index together when filtering or joining tables.

## Files in the data archive

| Path inside `data/` | Contents |
| --- | --- |
| `embeddings.npy` | Original `(146319, 768)` L2-normalized float32 vectors |
| `embedding_rows.csv` | Ordered PMID mapping, input-text hash, token count and quality flags |
| `papers.jsonl` | All 150,060 records: public bibliographic metadata, source links and available citation metrics |
| `neighbors.npz` | `indices` and `similarities` arrays for the 25 nearest papers per embedding row |
| `coordinates/display-{layout}-{2,3}d.npy` | Exact published coordinates for Original, Expanded and Spacious layouts |
| `coordinates/{layout}-{2,3}d.npy` | Selected numerical fits before display processing |
| `topics/assignments.csv` | Current broad/fine topic IDs and separately labeled legacy community IDs |
| `topics/hierarchy.json` | Topic names, descriptions, parent relations, keywords and representative paper IDs |
| `statistics/` | Existing publication, citation and concept summaries |
| `provenance.json`, `model.lock.json` | Source provenance, model identity and processing details |
| `manifest.json` | Exact file sizes, SHA-256 checksums, array shapes and alignment conventions |

The Spacious 2D display includes the accepted rigid rotation and five-decimal serialization. Use the `display-` arrays to reproduce the visible website. The unprefixed arrays retain numerical fits. The data manifest is the authoritative inventory for every file in this release.

To save a map figure:

```bash
python -m pip install matplotlib
python examples/plot_map.py --output atlas.png
```

## How it was produced

The corpus combines the original 100,000-record study, a 60-record coverage supplement, and a further 50,000 records selected to improve coverage. It is a frozen, coverage-directed PubMed collection, not a census of biomaterials research. The 3,741 unmapped records were retained despite embedding ineligibility from language or missing title. Among the embedded papers, 3,196 have title-only inputs and 5,516 inputs were truncated at the model limit.

The vectors use [AllenAI SPECTER2](https://huggingface.co/allenai/specter2): base revision `3447645e1def9117997203454fa4495937bfbd83`, proximity adapter revision `2081559630a80fc5851d8f798a05ba81e9468089`, title + tokenizer separator + abstract, a 512-token limit, first-token pooling, and float32 L2 normalization. The per-row input hashes identify the texts used; they are not the texts themselves.

The current hierarchy uses a weighted Louvain graph partition with connected-component refinement, Toponymy keyphrases and assistant-reviewed topic names. These exploratory labels aid navigation. They are not an expert-validated taxonomy. Older statistics also refer to **22 legacy communities**, which are preserved separately and should not be confused with the current 21 broad themes.

UMAP coordinates support visualization. Screen distances, apparent empty regions and topic counts do not measure scientific novelty, efficacy or the importance of a research problem. Use original-space vectors for semantic comparisons. Citation counts are dated accumulated counts; missing counts differ from zero. Sampling probabilities retained from earlier study components do not support pooled estimates of field prevalence across the full collection.

This release supports reuse of the frozen outputs. Reproducing the exact original acquisition and embedding run also requires the original source texts, pinned numerical environment and acquisition records. Publisher abstracts, raw source XML, model weights and private research notes are not included.

## Attribution and reuse

The original loading code and documentation in this repository are covered by the [MIT license](LICENSE). Project-created embeddings, coordinates, neighbor arrays and topic assignments/annotations are released under [CC BY 4.0](LICENSE-DATA.md), to the extent of the project's rights in those outputs. These grants do **not** relicense bibliographic metadata, pretrained models, third-party content or dependencies. See [NOTICE.md](NOTICE.md) for source attribution and the scope of the release.

Please cite the version you used; GitHub's **Cite this repository** menu reads [CITATION.cff](CITATION.cff). Also credit SPECTER2 and cite the original research papers supporting your work. There is no DOI assigned to this release.

## Checks

```bash
python -m unittest discover -s tests -v
python verify_release.py --data data
```

The first command checks the small loading and extraction examples without downloading the dataset. The second verifies the actual release files, paper-to-vector alignment, topic membership, vector normalization and sampled neighbor similarities.

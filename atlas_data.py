"""Load the published Biomaterials Atlas without loading embeddings into RAM."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np


class Atlas:
    """PMID-indexed access to a downloaded data release.

    Embedding rows are ordered by embedding_rows.csv. All coordinate arrays,
    topic assignments and neighbor indices use precisely this same order.
    Unmapped papers are present only in papers.jsonl.
    """

    def __init__(self, directory: str | Path = "data") -> None:
        self.directory = Path(directory)
        with (self.directory / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        with (self.directory / "embedding_rows.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["embedding_row"]) for row in rows] != list(range(len(rows))):
            raise ValueError("Embedding row IDs must be consecutive and ordered from zero")
        self.pmids = tuple(row["pmid"] for row in rows)
        self.row_by_pmid = {pmid: row for row, pmid in enumerate(self.pmids)}
        if len(self.row_by_pmid) != len(self.pmids):
            raise ValueError("Duplicate PMIDs in embedding row mapping")
        self.embeddings = np.load(self.directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if self.embeddings.shape != (len(self.pmids), self.manifest["counts"]["dimensions"]):
            raise ValueError("Embedding shape disagrees with row IDs or manifest")
        if self.embeddings.dtype != np.dtype("float32"):
            raise ValueError("Expected float32 embeddings")

    def papers(self) -> Iterator[dict]:
        """Iterate all mapped and unmapped bibliographic records."""
        with (self.directory / "papers.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def records(self, pmids: list[str]) -> dict[str, dict]:
        """Read selected records in one streaming pass through the metadata."""
        wanted = {str(pmid) for pmid in pmids}
        result = {}
        for paper in self.papers():
            if paper["pmid"] in wanted:
                result[paper["pmid"]] = paper
                if len(result) == len(wanted):
                    break
        return result

    def coordinates(self, layout: str = "spacious", dimensions: int = 2,
                    display: bool = True) -> np.ndarray:
        """Load row-aligned coordinates; display=True matches the public map."""
        if layout not in {"original", "expanded", "spacious"} or dimensions not in {2, 3}:
            raise ValueError("Choose original/expanded/spacious and 2 or 3 dimensions")
        prefix = "display-" if display else ""
        array = np.load(self.directory / "coordinates" / f"{prefix}{layout}-{dimensions}d.npy",
                        mmap_mode="r", allow_pickle=False)
        if array.shape != (len(self.pmids), dimensions):
            raise ValueError("Coordinate shape disagrees with embedding rows")
        return array

    def topic_assignments(self) -> list[dict[str, str]]:
        """Current L1 broad themes and L0 fine topics, plus legacy categories."""
        with (self.directory / "topics" / "assignments.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(self.pmids) or any(
            int(item["embedding_row"]) != row or item["pmid"] != self.pmids[row]
            for row, item in enumerate(rows)
        ):
            raise ValueError("Topic assignments disagree with embedding rows")
        return rows

    def related(self, pmid: str, k: int = 25, exact: bool = False) -> list[dict]:
        """Rank related papers by original-space cosine similarity.

        Use the stored exact top-25 neighbors by default. Setting exact=True,
        or requesting more than 25 results, scans the memory-mapped vectors.
        No UMAP distance or citation score enters this ranking.
        """
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k < len(self.pmids):
            raise ValueError(f"k must be an integer between 1 and {len(self.pmids) - 1}")
        pmid = str(pmid)
        if pmid not in self.row_by_pmid:
            raise KeyError(f"PMID {pmid} has no embedding in this release")
        row = self.row_by_pmid[pmid]
        if k <= 25 and not exact:
            with np.load(self.directory / "neighbors.npz", allow_pickle=False) as neighbors:
                indices = neighbors["indices"][row, :k].copy()
                similarities = neighbors["similarities"][row, :k].copy()
            if len(indices) != k or np.any(indices < 0) or np.any(indices >= len(self.pmids)) or row in indices:
                raise ValueError("Invalid precomputed neighbor indices")
        else:
            scores = np.empty(len(self.pmids), dtype=np.float32)
            query = self.embeddings[row]
            for start in range(0, len(scores), 4096):
                stop = min(start + 4096, len(scores))
                scores[start:stop] = self.embeddings[start:stop] @ query
            scores[row] = -np.inf
            # The row ID breaks equal-score ties deterministically.
            indices = np.lexsort((np.arange(len(scores)), -scores))[:k]
            similarities = scores[indices]
        records = self.records([self.pmids[int(index)] for index in indices])
        return [
            {**records[self.pmids[int(index)]], "embedding_row": int(index),
             "cosine_similarity": float(similarity)}
            for index, similarity in zip(indices, similarities)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find papers related to a PMID in Biomaterials Atlas")
    parser.add_argument("pmid")
    parser.add_argument("--data", default="data", help="Extracted data directory")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exact", action="store_true", help="Recompute cosine ranking from the stored vectors")
    args = parser.parse_args()
    atlas = Atlas(args.data)
    for paper in atlas.related(args.pmid, args.k, args.exact):
        print(f"{paper['cosine_similarity']:.5f}\t{paper['pmid']}\t{paper.get('title', '')}")


if __name__ == "__main__":
    main()

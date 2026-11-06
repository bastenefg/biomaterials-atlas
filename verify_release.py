"""Verify integrity and meaningful alignment checks on an extracted release."""
import argparse
import json

import numpy as np

from atlas_data import Atlas
from download_data import verify_data


def verify(directory):
    manifest = verify_data(directory)
    atlas = Atlas(directory)
    counts = manifest["counts"]
    if len(atlas.pmids) != counts["embedded"]:
        raise ValueError("Embedding count differs from the release manifest")
    seen = set()
    unmapped = 0
    for number, paper in enumerate(atlas.papers()):
        pmid = paper["pmid"]
        if not isinstance(pmid, str) or pmid in seen:
            raise ValueError("PMIDs must be unique strings")
        seen.add(pmid)
        row = paper["embedding_row"]
        if row is None:
            unmapped += 1
        elif row != number or atlas.pmids[row] != pmid:
            raise ValueError("Metadata order or embedding row identity mismatch")
    if len(seen) != counts["records"] or unmapped != counts["unmapped"]:
        raise ValueError("Metadata counts disagree with the release manifest")
    maximum_norm_error = 0.0
    for start in range(0, len(atlas.pmids), 4096):
        block = atlas.embeddings[start:start + 4096]
        if not np.isfinite(block).all():
            raise ValueError("Embedding array contains a nonfinite value")
        maximum_norm_error = max(maximum_norm_error, float(np.max(np.abs(np.linalg.norm(block, axis=1) - 1))))
    if maximum_norm_error > 1e-5:
        raise ValueError("Vectors are not L2 normalized")
    topics = atlas.topic_assignments()
    if len({item["broad_topic"] for item in topics}) != counts["broad_topics"]:
        raise ValueError("Broad-topic count mismatch")
    if len({item["fine_topic"] for item in topics}) != counts["fine_topics"]:
        raise ValueError("Fine-topic count mismatch")
    for layout in ("original", "expanded", "spacious"):
        for dimensions in (2, 3):
            for display in (False, True):
                if not np.isfinite(atlas.coordinates(layout, dimensions, display)).all():
                    raise ValueError("Coordinates contain nonfinite values")
    with np.load(atlas.directory / "neighbors.npz", allow_pickle=False) as neighbors:
        indices = neighbors["indices"]
        similarities = neighbors["similarities"]
    if indices.shape != (len(atlas.pmids), 25) or similarities.shape != indices.shape:
        raise ValueError("Neighbor array shape mismatch")
    if np.any(indices < 0) or np.any(indices >= len(atlas.pmids)) or np.any(indices == np.arange(len(indices))[:, None]):
        raise ValueError("Neighbor indices are out of bounds or contain self matches")
    if not np.isfinite(similarities).all() or np.any(np.diff(similarities, axis=1) > 1e-6):
        raise ValueError("Neighbor similarities are nonfinite or incorrectly ranked")
    checked_rows = np.linspace(0, len(atlas.pmids) - 1, 17, dtype=int)
    maximum_cosine_error = 0.0
    for row in checked_rows:
        actual = atlas.embeddings[indices[row]] @ atlas.embeddings[row]
        maximum_cosine_error = max(maximum_cosine_error, float(np.max(np.abs(actual - similarities[row]))))
    if maximum_cosine_error > 1e-5:
        raise ValueError("Stored neighbor similarities disagree with the original vectors")
    return {"status": "passed", "files": len(manifest["files"]), "records": len(seen),
            "embeddings": len(atlas.pmids), "unmapped": unmapped,
            "maximum_norm_error": maximum_norm_error, "sampled_neighbor_rows": len(checked_rows),
            "maximum_cosine_error": maximum_cosine_error}


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(verify(args.data), indent=2))

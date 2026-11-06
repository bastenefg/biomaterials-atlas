import csv
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

import numpy as np

from atlas_data import Atlas
from download_data import safe_extract, verify_data


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "source" / "data"
        self.data.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        files = [{"path": item.relative_to(self.data).as_posix(), "bytes": item.stat().st_size,
                  "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                 for item in self.data.rglob("*") if item.is_file() and item.name != "manifest.json"]
        (self.data / "manifest.json").write_text(json.dumps({"files": files, "counts": {"dimensions": 2}}), encoding="utf-8")

    def test_cosine_and_metadata_identity(self):
        vectors = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype="float32")
        np.save(self.data / "embeddings.npy", vectors)
        with (self.data / "embedding_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["embedding_row", "pmid"])
            writer.writerows(enumerate(["101", "102", "103", "104"]))
        with (self.data / "papers.jsonl").open("w", encoding="utf-8") as handle:
            for row, pmid in enumerate(["101", "102", "103", "104", "999"]):
                handle.write(json.dumps({"pmid": pmid, "embedding_row": row if row < 4 else None,
                                         "title": f"Paper {pmid}"}) + "\n")
        scores = vectors @ vectors.T
        np.fill_diagonal(scores, -np.inf)
        indices = np.argsort(-scores, axis=1)[:, :3]
        np.savez(self.data / "neighbors.npz", indices=indices, similarities=np.take_along_axis(scores, indices, axis=1))
        self.manifest()
        atlas = Atlas(self.data)
        self.assertIsInstance(atlas.embeddings, np.memmap)
        for exact in (False, True):
            related = atlas.related("101", k=2, exact=exact)
            self.assertEqual([paper["pmid"] for paper in related], ["102", "103"])
            self.assertAlmostEqual(related[0]["cosine_similarity"], 0.8, places=6)
            self.assertEqual(related[0]["title"], "Paper 102")
        with self.assertRaisesRegex(KeyError, "no embedding"):
            atlas.related("999", k=2)
        with self.assertRaises(ValueError):
            atlas.related("101", k=0)

    def test_verify_then_reject_corruption(self):
        (self.data / "example.txt").write_text("valid", encoding="utf-8")
        self.manifest()
        verify_data(self.data)
        (self.data / "example.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            verify_data(self.data)

    def test_safe_extract_and_traversal_rejection(self):
        (self.data / "example.txt").write_text("valid", encoding="utf-8")
        self.manifest()
        archive = self.root / "valid.zip"
        with zipfile.ZipFile(archive, "w") as package:
            for item in self.data.iterdir():
                package.write(item, "data/" + item.name)
        stage = self.root / "staged"
        stage.mkdir()
        self.assertEqual((safe_extract(archive, stage) / "example.txt").read_text(), "valid")
        for bad in ["../escape", "data/../../escape", "data/C:escape", "data\\escape", "/data/escape"]:
            with self.subTest(path=bad):
                malicious = self.root / "bad.zip"
                with zipfile.ZipFile(malicious, "w") as package:
                    package.writestr(bad, "bad")
                with self.assertRaises(ValueError):
                    safe_extract(malicious, stage)

    def test_symlink_rejected(self):
        archive = self.root / "symlink.zip"
        entry = zipfile.ZipInfo("data/link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr(entry, "../../escape")
        with self.assertRaisesRegex(ValueError, "nonregular"):
            safe_extract(archive, self.root)


if __name__ == "__main__":
    unittest.main()

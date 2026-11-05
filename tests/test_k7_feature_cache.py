"""Fast K7 source tensor cache and no-reencoding regression tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from urban_transfer.k7_feature_cache import build_tensor_cache, read_tensor_cache
from urban_transfer.m3_features import FeatureStore
from urban_transfer.protocol import SOURCE_TASKS


class K7TensorCacheTests(unittest.TestCase):
    def test_unique_tile_cache_and_stale_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image_root = root/"images"
            image_root.mkdir()
            feature_root = root/"features"
            store = FeatureStore(feature_root, encoder_id="RemoteCLIP-ViT-B-32")
            rows, entries = [], []
            for i, city in enumerate(("A", "B", "C")):
                path = image_root/f"{city}.png"
                Image.fromarray(np.full((4, 4, 3), 30+i, dtype=np.uint8)).save(path)
                entries.append((path, torch.full((8,), float(i+1))))
                for task in SOURCE_TASKS:
                    rows.append({
                        "city": city, "image_name": path.name,
                        "indicator_column": task,
                        "patch_lon": 4.0, "patch_lat": 12.0,
                        "patch_area_km2": 2.0,
                        "indicator_value_log1p": 1.0+i,
                    })
            store.put_many(entries)
            csv_path = root/"train.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            target = root/"source_tensors.pt"
            build_tensor_cache(csv_path, image_root, feature_root, "ViT-B-32", target)
            first = target.stat().st_mtime_ns
            build_tensor_cache(csv_path, image_root, feature_root, "ViT-B-32", target)
            self.assertEqual(first, target.stat().st_mtime_ns)
            frame, x, context, y, names = read_tensor_cache(
                target, csv_path, image_root, feature_root, "ViT-B-32")
            self.assertEqual(len(frame), 3*len(SOURCE_TASKS))
            self.assertEqual(tuple(x.shape), (len(frame), 8))
            self.assertEqual(tuple(context.shape), (len(frame), 4))
            self.assertEqual(tuple(y.shape), (len(frame),))
            self.assertEqual(names, frame.indicator_column.astype(str).tolist())
            self.assertTrue(torch.allclose(x[0], x[1]))
            # A modified source table must invalidate the source-only cache.
            with csv_path.open("a") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                read_tensor_cache(target, csv_path, image_root, feature_root, "ViT-B-32")


if __name__ == "__main__":
    unittest.main()

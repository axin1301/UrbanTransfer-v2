"""Shared CSV split defaults for M2 and M3.

Explicit --*-csv arguments always override directory-derived defaults.
"""
import os
from pathlib import Path

DEFAULT_SPLIT_DIR = "/scratch/project_462001391/xiyanxin/Urban_experience/codes_BaseStation/city_indicator_splits"
SPLIT_FILES = {
    "train_csv": "train.csv",
    "scui_csv": "seen_city_test.csv",
    "ucsi_csv": "unseen_city_seen_indicator_test.csv",
    "ucui_csv": "unseen_city_test.csv",
}

def add_data_dir_argument(parser):
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("URBAN_SPLIT_DIR", DEFAULT_SPLIT_DIR),
        help="Directory containing train/SCUI/UCSI/UCUI CSVs; overridden by individual --*-csv options",
    )

def resolve_data_paths(args):
    """Populate absent CSV paths, preserving explicitly provided filenames."""
    root = Path(args.data_dir).expanduser()
    for name, filename in SPLIT_FILES.items():
        if not getattr(args, name, None):
            setattr(args, name, str(root / filename))
    return args

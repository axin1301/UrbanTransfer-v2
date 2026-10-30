import argparse
import os
import unittest
from unittest.mock import patch
from urban_transfer.data_paths import (
    DEFAULT_SPLIT_DIR, SPLIT_FILES, add_data_dir_argument, resolve_data_paths
)

class DataPathTests(unittest.TestCase):
    def parser(self):
        p = argparse.ArgumentParser()
        add_data_dir_argument(p)
        for name in SPLIT_FILES:
            p.add_argument('--' + name.replace('_', '-'))
        return p

    def test_defaults(self):
        args = resolve_data_paths(self.parser().parse_args([]))
        for name, filename in SPLIT_FILES.items():
            self.assertEqual(getattr(args, name), DEFAULT_SPLIT_DIR + '/' + filename)

    def test_custom_directory_and_explicit_override(self):
        args=resolve_data_paths(self.parser().parse_args(
            ['--data-dir','/other/splits','--scui-csv','/special/scui.csv']
        ))
        self.assertEqual(args.train_csv,'/other/splits/train.csv')
        self.assertEqual(args.scui_csv,'/special/scui.csv')

    def test_env_override(self):
        with patch.dict(os.environ, {'URBAN_SPLIT_DIR':'/env/splits'}):
            args=resolve_data_paths(self.parser().parse_args([]))
        self.assertEqual(args.ucui_csv,'/env/splits/unseen_city_test.csv')

if __name__=='__main__':
    unittest.main()

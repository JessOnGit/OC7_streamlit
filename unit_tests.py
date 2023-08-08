import unittest
import os

import pandas as pd
import json


class TestFolderContents(unittest.TestCase):

    def test_refs_folder_contents(self):
        expected_files = ['min_max_mean_med.csv', 'scores0.csv', 'scores1.csv', 'shap_explainer.sav',
                          'shap_summary_plot.png', 'data_columns.json']

        # list of files in 'ref'
        actual_files = os.listdir('refs')

        # check if needed files in 'ref'
        for file in expected_files:
            with self.subTest(file=file):
                self.assertIn(file, actual_files, f"File '{file}' not found in 'refs' folder.")

    def test_data_files(self):
        data_folder = 'data_test_clean'
        x = 'X_test_sample.csv'
        y = 'y_test_sample.csv'

        # Check if X_test_sample.csv and y_test_sample.csv are in data_test_clean folder
        self.assertTrue(os.path.exists(os.path.join(data_folder, x)))
        self.assertTrue(os.path.exists(os.path.join(data_folder, y)))

        # Check if X_test_sample.csv and y_test_sample.csv have the same number of columns (x)
        x_test_path = os.path.join(data_folder, x)
        y_test_path = os.path.join(data_folder, y)
        x_test_df = pd.read_csv(x_test_path)
        y_test_df = pd.read_csv(y_test_path)
        x_samples = x_test_df.shape[0]
        y_samples = y_test_df.shape[0]
        self.assertEqual(x_samples, y_samples)

        # Check if X_test_sample.csv has 1060 rows
        expected_rows = 591
        actual_rows = x_test_df.shape[1]
        self.assertEqual(actual_rows, expected_rows)

        # Check if y_test_sample.csv has 1 row
        expected_y_rows = 1
        actual_y_rows = y_test_df.shape[1]
        self.assertEqual(actual_y_rows, expected_y_rows)

        # Chek if columns names are right
        columns = x_test_df.columns.tolist()
        columns.append(y_test_df.columns.tolist()[0])
        with open('refs/data_columns.json', 'r') as json_file:
            ref_columns = json.load(json_file)
        self.assertEqual(ref_columns, columns)
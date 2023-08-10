import unittest
import os
import pandas as pd
import json
# from utils import get_median_scores


# SubFunctions
def are_different(num1, num2):
    return num1 != num2


def is_scalar(value):
    return isinstance(value, (int, float))


def has_different_values(df):
    if isinstance(df, pd.DataFrame):
        return df.nunique().gt(1).any()
    else:
        return False


# UnitTest Folder contents class
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


# Copy version of get_median_scores from utils (cached in streamlit for the app
# >> impossible to import without having a streamlit conflict)
def get_median_scores2():
    sc0 = pd.read_csv('refs/scores0.csv')
    sc1 = pd.read_csv('refs/scores1.csv')
    median_sc0 = sc0.median()[0]
    median_sc1 = sc1.median()[0]
    return sc0, sc1, median_sc0, median_sc1


# UnitTest functions testing class
class TestUtilsFunctions(unittest.TestCase):

    def test_median_scoring(self):
        sc0, sc1, median_sc0, median_sc1 = get_median_scores2()

        # Check sc0 and sc1 are lists and both lists contain various values
        self.assertTrue(has_different_values(sc0))
        self.assertTrue(has_different_values(sc1))

        # Check that both median are scalars
        self.assertTrue(is_scalar(median_sc0))
        self.assertTrue(is_scalar(median_sc1))

        # Check that medians are different from one another
        self.assertTrue(are_different(median_sc0, median_sc1))


if __name__ == "__main__":
     unittest.main()
import tempfile
import unittest
from pathlib import Path
import numpy as np
from generate_data import generate_dataset
from freshflow import load_data, validate_data, calculate_metrics, run_analysis, daily_summary

class FreshFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.csv = Path(self.temp.name) / "sales.csv"
        generate_dataset(self.csv)

    def tearDown(self):
        self.temp.cleanup()

    def test_dataset_has_5000_rows_and_12_columns(self):
        data = load_data(self.csv)
        self.assertEqual(data.shape, (5000, 12))

    def test_dataset_covers_all_30_days(self):
        data = load_data(self.csv)
        self.assertTrue(np.array_equal(np.unique(data[:, 1]), np.arange(1, 31)))

    def test_missing_values_are_loaded_as_nan(self):
        data = load_data(self.csv)
        self.assertEqual(int(np.isnan(data[:, 6]).sum()), 4)
        self.assertEqual(int(np.isnan(data[:, 11]).sum()), 25)

    def test_validation_flags_bad_rows_but_keeps_rating_only_issues(self):
        data = load_data(self.csv)
        flags, reasons, valid = validate_data(data)
        self.assertTrue(flags[10])
        self.assertIn("invalid_unit_price", reasons[70])
        self.assertIn("missing_customer_rating", reasons[30])
        self.assertTrue(valid[30])

    def test_metric_formulas(self):
        sample = np.array([[1, 1, 1, 1, 1, 2, 100, 40, .1, 10, 1, 5]], dtype=float)
        rows, metrics = calculate_metrics(sample, np.array([True]))
        self.assertEqual(metrics["gross_sales"][0], 200)
        self.assertAlmostEqual(metrics["discount_amount"][0], 20)
        self.assertAlmostEqual(metrics["net_revenue"][0], 180)
        self.assertAlmostEqual(metrics["cost_of_goods_sold"][0], 80)
        self.assertAlmostEqual(metrics["gross_profit"][0], 100)


    def test_daily_trend_contains_changes_and_rolling_average(self):
        data = load_data(self.csv)
        _, metrics = calculate_metrics(data, np.ones(len(data), dtype=bool))
        daily = daily_summary(data, metrics)
        self.assertEqual(daily.shape, (30, 8))
        self.assertTrue(np.isnan(daily[0, 4]))
        self.assertTrue(np.isfinite(daily[6, 6]))

    def test_invalid_column_shape_is_rejected(self):
        bad_file = Path(self.temp.name) / "bad.csv"
        bad_file.write_text("a,b,c" + chr(10) + "1,2,3" + chr(10))
        with self.assertRaises(ValueError):
            load_data(bad_file)

    def test_full_workflow_writes_report_and_outputs(self):
        out = Path(self.temp.name) / "outputs"
        report = run_analysis(self.csv, out)
        self.assertEqual(report["dataset_profile"]["rows_loaded"], 5000)
        self.assertTrue((out / "final_report.json").exists())
        self.assertTrue((out / "quality_report.csv").exists())
        self.assertTrue((out / "daily_trend.csv").exists())
        self.assertTrue((out / "anomaly_register.csv").exists())

if __name__ == "__main__":
    unittest.main()

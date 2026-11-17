"""Run with: python -m unittest -v test_portfolio.py"""
import unittest
import numpy as np
from portfolio_analysis import demo_returns, asset_moments, optimize_portfolios, forecast_expanding, forecast_metrics, ASSETS


class PortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.returns = demo_returns('2016-01','2025-12')
        cls.mu, cls.cov = asset_moments(cls.returns)
        cls.weights, cls.frontier = optimize_portfolios(cls.mu, cls.cov, .02)

    def test_dimensions_and_no_lookahead(self):
        self.assertEqual(self.returns.shape,(120,12))
        self.assertEqual(len(self.dates),120)
        self.assertEqual(self.cov.shape,(12,12))
        self.assertTrue(np.allclose(self.cov,self.cov.T))

    def test_feasible_portfolios(self):
        for w in self.weights.values():
            self.assertAlmostEqual(float(w.sum()),1.,places=6)
            self.assertTrue(np.all(w >= -1e-8))
            self.assertTrue(np.all(w <= 1+1e-8))

    def test_frontier_and_min_variance(self):
        self.assertEqual(len(self.frontier),45)
        self.assertTrue(all(x['volatility'] >= 0 for x in self.frontier))
        self.assertAlmostEqual(self.frontier[0]['volatility'],
                               float(np.sqrt(self.weights['minvar']@self.cov@self.weights['minvar'])),places=6)

    def test_walk_forward_predictions(self):
        f=forecast_expanding(self.returns, ASSETS.index('BusEq'),24)
        self.assertEqual(len(f),24)
        self.assertEqual(f[0]['month_index'],96)
        self.assertTrue(all(row['random_walk_pred']==0 for row in f))
        self.assertGreater(forecast_metrics(f)['ar1_rmse'],0)


if __name__=='__main__':
    unittest.main()

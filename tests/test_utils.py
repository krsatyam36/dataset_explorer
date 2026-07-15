"""Tests for utility classes."""

import time
from datasets_explorer.utils import RateLimiter


class TestRateLimiter:
    def test_default_delay(self):
        limiter = RateLimiter()
        t0 = time.time()
        limiter.wait("default")
        t1 = time.time()
        assert t1 - t0 < 0.1  # first call should not block

    def test_second_call_blocks(self):
        limiter = RateLimiter()
        limiter.wait("default")
        t0 = time.time()
        limiter.wait("default")
        t1 = time.time()
        assert t1 - t0 >= 0.5  # should have waited close to default delay (1.5s)

    def test_different_domains_independent(self):
        limiter = RateLimiter()
        limiter.wait("domain-a.com")
        t0 = time.time()
        limiter.wait("domain-b.com")
        t1 = time.time()
        assert t1 - t0 < 0.5  # different domain, no significant wait

    def test_url_parsing(self):
        limiter = RateLimiter()
        limiter.wait("https://kaggle.com/datasets/test")
        t0 = time.time()
        limiter.wait("https://www.kaggle.com/datasets/other")
        t1 = time.time()
        assert t1 - t0 >= 2.0  # kaggle delay is 3.0

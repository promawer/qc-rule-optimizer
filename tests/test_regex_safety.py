import unittest

from app.regex_safety import RegexTimeout, safe_search


class RegexSafetyTest(unittest.TestCase):
    def test_runaway_pattern_is_killed_and_next_request_recovers(self):
        first = safe_search("验证码", "请勿泄露验证码")
        self.assertEqual(first.group(0), "验证码")
        with self.assertRaises(RegexTimeout):
            safe_search("(a+)+$", "a" * 10000 + "!", timeout=.05)
        next_result = safe_search("投诉", "客户投诉")
        self.assertEqual(next_result.span(), (2, 4))


if __name__ == "__main__":
    unittest.main()

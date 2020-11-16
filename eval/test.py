#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest

from evaluator import regex


class RegexTest(unittest.TestCase):

    def test_parse(self):
        submission = '홍길동_180000_assignsubmission_file_'
        self.assertEqual(regex.search(submission).group('name'), '홍길동')
        submission = '김수한무_181234_assignsubmission_file_'
        self.assertEqual(regex.search(submission).group('name'), '김수한무')
        submission = 'John Doe_181234_assignsubmission_file_'
        self.assertEqual(regex.search(submission).group('name'), 'John Doe')
        submission = 'something'
        self.assertEqual(regex.search(submission), None)


if __name__ == "__main__":
    unittest.main()

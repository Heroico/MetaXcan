#!/usr/bin/env python
import unittest
import sys
import shutil
import os
import re
import gzip

if "DEBUG" in sys.argv:
    sys.path.insert(0, "..")
    sys.path.insert(0, "../../")
    sys.path.insert(0, ".")
    sys.argv.remove("DEBUG")

from metax.DataSet import DataSet
from metax.DataSet import DataSetFileUtilities
from metax.DataSet import DataSetCollection
import metax.Formats as Formats

from M00_prerequisites import ProcessPrerequisites

class Dummy(object):
    pass

def buildDummyArgs(root):
    dummy = Dummy()
    dummy.verbosity = 10
    dummy.dosage_folder = os.path.join(root, "dosage_set_1")
    dummy.snp_list = os.path.join(root, "snp.txt.gz")
    dummy.output_folder = os.path.join(root, "intermediate/filtered")
    dummy.file_pattern = "set_(.*)"
    dummy.population_group_filters = ["HERO"]
    dummy.individual_filters = ["ID.*"]
    dummy.input_format = Formats.IMPUTE
    dummy.output_format = Formats.PrediXcan
    return dummy

def setupDataForArgs(args, root):
    if os.path.exists(root):
        shutil.rmtree(root)
    shutil.copytree("tests/_td", root)

def cleanUpDataForArgs(root):
    shutil.rmtree(root)

class TestM00(unittest.TestCase):

    def testProcessPrerequisitesnoArgConstructor(self):
        with self.assertRaises(AttributeError):
            dummy = Dummy()
            p = ProcessPrerequisites(dummy)

    def testProcessPrerequisitesConstructor(self):
        dummy = buildDummyArgs("_test")
        setupDataForArgs(dummy, "_test")
        p = ProcessPrerequisites(dummy)
        self.assertEqual(p.dosage_folder, "_test/dosage_set_1")
        self.assertEqual(p.snp_list, "_test/snp.txt.gz")
        self.assertEqual(p.output_folder, "_test/intermediate/filtered")
        self.assertEqual(p.population_group_filters, ["HERO"])
        self.assertEqual(p.individual_filters, [re.compile("ID.*")])
        self.assertEqual(p.chromosome_in_name_regex,re.compile("set_(.*)"))
        self.assertEqual(p.samples_input, "_test/dosage_set_1/set.sample")
        self.assertEquals(p.samples_output, "_test/intermediate/filtered/set.sample")
        cleanUpDataForArgs("_test")

    def testProcessPrerequisitesRun(self):
        dummy = buildDummyArgs("_test")
        setupDataForArgs(dummy, "_test")
        p = ProcessPrerequisites(dummy)

        try:
            p.run()
        except:
            self.assertEqual(False, True, "Prerequisites should have run without error")

        with open(p.samples_output) as f:
            expected_lines = ["ID POP GROUP SEX",
                              "ID1 K HERO male",
                              "ID2 K HERO female",
                              "ID3 K HERO female"]
            for i,expected_line in enumerate(expected_lines):
                actual_line = f.readline().strip()
                self.assertEqual(actual_line, expected_line)

        path = os.path.join(p.output_folder, "set_chr1.dosage.gz")
        with gzip.open(path) as f:
            expected_lines = ["chr1 rs2 2 G A 0.166666666667 0 1 0",
                              "chr1 rs3 3 G A 0.333333333333 0 1 1",
                              "chr1 rs4 4 G A 1.0 2 2 2"]
            for i,expected_line in enumerate(expected_lines):
                actual_line = f.readline().strip()
                self.assertEqual(actual_line, expected_line)
        cleanUpDataForArgs("_test")

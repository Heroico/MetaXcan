import numpy
import numpy.testing
import pandas

import unittest

from metax import Exceptions
from metax import MatrixManager
from metax import MatrixManager2
from metax.genotype import GeneExpressionMatrixManager
from metax import PredictionModel

def prerequisites(MatrixManagerKlass=MatrixManager.MatrixManager, preprocess_models=None):
    model_manager = PredictionModel.load_model_manager("tests/_td/dbs_3/", preprocess_models=preprocess_models)
    snp_covariance = pandas.read_table("tests/_td/meta_covariance/snps_covariance.txt.gz")
    gene_expression_manager = GeneExpressionMatrixManager.GeneExpressionMatrixManager(snp_covariance, model_manager, MatrixManagerKlass=MatrixManagerKlass)
    return model_manager, gene_expression_manager, snp_covariance

def _reduce_models(models):
    snps_ = {x for x in models.loc["ENSG00000107937.14"].loc["Muscle_Skeletal"].index if x in models.loc["ENSG00000107937.14"].loc["Adipose_Subcutaneous"].index} | \
            set(models.loc["ENSG00000107937.14"].loc["Muscle_Skeletal"].index[0:3]) | \
            set(models.loc["ENSG00000107937.14"].loc["Adipose_Subcutaneous"].index[0:3]) | \
            set(models.loc["ENSG00000107937.14"].loc["Whole_Blood"].index[0:3])

    idx = pandas.IndexSlice
    models = models.loc[idx["ENSG00000107937.14", ["Muscle_Skeletal", "Adipose_Subcutaneous", "Whole_Blood"], snps_],:]
    return snps_, models

#Order of rsids is different than the funciton above so that selection is in all effect different
def _reduce_models_preprocess(models):
    m_ = models.loc[(models.gene == "ENSG00000107937.14") & models.model.isin({"Muscle_Skeletal", "Adipose_Subcutaneous", "Whole_Blood"})]
    snps_ = set(m_.loc[m_.model == "Muscle_Skeletal"].rsid).intersection(set(m_.loc[m_.model == "Adipose_Subcutaneous"].rsid)) | \
            set(models.loc[models.model == "Muscle_Skeletal"].rsid.values[0:3]) | \
            set(models.loc[models.model == "Adipose_Subcutaneous"].rsid.values[0:3]) | \
            set(models.loc[models.model == "Whole_Blood"].rsid.values[0:3])

    m_ = m_.loc[m_.rsid.isin(snps_)]
    return m_

def _reduced_test(unit_test, gene_expression_manager):
    snps, matrix = gene_expression_manager.get("ENSG00000107937.14",
                                               ["Muscle_Skeletal", "Adipose_Subcutaneous", "Whole_Blood"])
    e_snps = ['Adipose_Subcutaneous', 'Muscle_Skeletal', 'Whole_Blood']
    e_matrix = [[ 1.        ,  0.80567325, -0.04308622],
                [ 0.80567325,  1.        , -0.016737  ],
                [-0.04308622, -0.016737  ,  1.        ]]
    unit_test.assertEqual(snps, e_snps)
    numpy.testing.assert_array_almost_equal(matrix, e_matrix)

class TestGeneExpressionMatrixManager(unittest.TestCase):
    def test_gene_expression_matrix_manager_1(self):
        model_manager, gene_expression_manager, snp_covariance = prerequisites(preprocess_models=_reduce_models_preprocess)
        snps_, models = _reduce_models(model_manager.models)

        _reduced_test(self, gene_expression_manager)

    def test_gene_expression_matrix_manager_2(self):
        model_manager, gene_expression_manager, snp_covariance = prerequisites(MatrixManagerKlass=MatrixManager2.MatrixManager2, preprocess_models=_reduce_models_preprocess)

        _reduced_test(self, gene_expression_manager)
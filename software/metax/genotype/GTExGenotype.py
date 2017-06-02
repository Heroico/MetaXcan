import numpy

class GF(object):
    RSID=0
    CHROMOSOME=1
    POSITION=2
    REF_ALLELE=3
    ALT_ALLELE=4
    FREQUENCY=5
    FIRST_DOSAGE=6

from ..misc import KeyedDataSource
from .. import Utilities
import gzip
import logging

def parse_gtex_variant(variant):
    comps = variant.split("_")
    return comps[0:4]

def gtex_geno_header(gtex_file):
    with gzip.open(gtex_file) as file:
        header = file.readline().strip().split()
    return header

def gtex_geno_lines(gtex_file, gtex_snp_file, snps=None):
    logging.log(9, "Loading GTEx snp file")
    gtex_snp = KeyedDataSource.load_data(gtex_snp_file, "VariantID", "RS_ID_dbSNP142_CHG37p13", numeric=False)

    logging.log(9, "Processing GTEx geno")
    with gzip.open(gtex_file) as file:
        for i,line in enumerate(file):
            if i==0:continue #skip header. This line is not needed but for conceptual ease of mind
            comps = line.strip().split()
            variant = comps[0]

            if not variant in gtex_snp:
                continue

            rsid = gtex_snp[variant]
            if snps and not rsid in snps:
                continue

            data = parse_gtex_variant(variant)
            dosage = numpy.array(comps[1:], dtype=numpy.float64)
            frequency = numpy.mean(dosage)/2

            yield [rsid] + data + [frequency] + list(dosage)

def gtex_geno_by_chromosome(gtex_file, gtex_snp_file, snps=None):
    buffer = []
    last_chr = None

    def _buffer_to_data(buffer):
        _data = zip(*buffer)

        _metadata = _data[0:GF.FIRST_DOSAGE]
        rsids = _metadata[GF.RSID]
        chromosome = int(_metadata[GF.CHROMOSOME][0]) # I know I am gonna regret this cast
        _metadata = zip(*_metadata)
        metadata = Utilities.to_dataframe(_metadata, ["rsid", "chromosome", "position", "ref_allele", "alt_allele", "frequency"], to_numeric="ignore")

        dosage = zip(*_data[GF.FIRST_DOSAGE:])
        dosage_data = {}
        #TODO: 142_snps are not unique, several rows go into the same value, improve this case handling
        for i in xrange(0, len(rsids)):
            rsid = rsids[i]
            if rsid in dosage_data:
                ind = metadata[metadata.rsid == rsid].index.tolist()[0]
                metadata = metadata.drop(ind)

            dosage_data[rsid] = dosage[i]
        metadata["number"] = range(0, len(metadata))
        metadata = metadata.set_index("number")

        return chromosome, metadata, dosage_data

    logging.log(8, "Starting to process lines")
    for line in gtex_geno_lines(gtex_file, gtex_snp_file, snps):

        chromosome = line[GF.CHROMOSOME]
        if last_chr is None: last_chr = chromosome

        if last_chr != chromosome:
            yield _buffer_to_data(buffer)
            buffer = []

        last_chr = chromosome
        buffer.append(line)

    yield _buffer_to_data(buffer)
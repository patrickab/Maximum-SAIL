from ribs.archives import GridArchive, ArchiveDataFrame
from pandas import DataFrame
import pandas

from config.config import Config
config = Config('config/config.ini')
SOL_DIMENSION = config.SOL_DIMENSION
OBJ_BHV_NUMBER_BINS = config.OBJ_BHV_NUMBER_BINS
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE

def csv_to_archive(csv_path: str):

    min_obj_threshhold = -1

    df = pandas.read_csv(csv_path)
    df = ArchiveDataFrame(df)

    archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=OBJ_BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = min_obj_threshhold
    )

    archive.add(df.solution_batch(), df.objective_batch(), df.measures_batch())
    return archive


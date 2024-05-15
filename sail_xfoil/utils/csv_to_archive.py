from ribs.archives import GridArchive, ArchiveDataFrame
import pandas

SOL_DIMENSION = 11
OBJ_BHV_NUMBER_BINS = [25,25]
BHV_VALUE_RANGE = [(0.2625,0.6875), (0.0725,0.1875)]

def csv_to_archive(csv_path: str):

    df = pandas.read_csv(csv_path)
    df = ArchiveDataFrame(df)

    archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=OBJ_BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
    )

    archive.add(df.solution_batch(), df.objective_batch(), df.measures_batch())
    return archive


from ribs.archives import GridArchive, ArchiveDataFrame
from pandas import DataFrame
import pandas

df = pandas.read_csv('exceptional_obj_archive_hybrid_verification_ucb_init_mes_0_51600.csv').sort_values(by=['objective'], ascending=False)
df = ArchiveDataFrame(df)

fufu = 5
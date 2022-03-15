import wget
import datetime
import pandas as pd

dti = pd.date_range("2022-02-09", periods=130, freq="D")
dti = pd.to_datetime(dti).date
filename = 'https://dsc-mp2022.s3.amazonaws.com/proj-b/dataset-b01/Tst'
filepostfix = 'LOBs.txt'
output_directory = 'E:/mini_proj_data/HSBC_raw_LOBs'

for date in dti:
    filepath = filename + str(date) + filepostfix
    try:
        wget.download(filepath, out=output_directory)
    except:
        print(date)

import glob
import os
from datetime import datetime

import pandas as pd


def get_key():
    """
    Retrives API key from local file.
    """
    with open("/Users/tianyudu/Documents/UToronto/Course/ECO499/stock_news_api.txt") as f:
        key = f.readline()
        key = key.replace("\n", "")
    return key


def gen_call(
    api_key: str,
    start_date: str,
    end_date: str,
    page: str
) -> str:
    """
    Generates API call.
    """
    call = "https://stocknewsapi.com/api/v1/category?\
section=general\
&items=50\
&page={page}\
&datatype=csv\
&date={sd}-{ed}\
&type=article\
&sortby=trending\
&token={key}"
    return call.format(
        page=page,
        sd=start_date,
        ed=end_date,
        key=api_key
    )


def batch_download(
    start: str,
    end: str,
    num_pages: int = 1,
):
    key = get_key()
    os.system("pwd")
    if not os.path.exists("./data/batch_download"):
        os.system("mkdir ./data/batch_download")
    for date in pd.date_range(start=start, end=end):
        t = date.strftime(format="%m%d%Y")
        for p in range(num_pages):
            file_name = "{}_page{}.csv".format(t, p)
            # print(gen_call(api_key=key, start_date=t, end_date=t, page=str(p)))
            os.system("wget --output-document='./data/batch_download/{}' '{}'".format(
                file_name, gen_call(api_key=key, start_date=t, end_date=t, page=str(p))
            ))


def concanate_files(
    path: str = "./data/batch_download",
    dest: str = "./data"
) -> pd.DataFrame:
    files = sorted(glob.glob("{}/*.csv".format(path)))
    df = pd.concat(map(pd.read_csv, files))
    df.reset_index(drop=True, inplace=True)
    df.to_csv(dest + "/ds.csv")
    return df


from contextlib import contextmanager
import pandas as pd
import json
import time
import itertools

@contextmanager
def mytimer():
    t0 = time.time()
    try:
        yield t0
    finally:
        t1 = time.time()
        print(t1-t0)

nrow = 10_000_000

with mytimer():
    row = ['%313460000000001','%Google LLC',12345678,87654321]
    row_list = list(itertools.repeat(row, nrow))
    # row_byte = json.dumps(row_list)

# with mytimer():
#     json.loads(row_byte)

with mytimer():
    df = pd.DataFrame(row_list)
    # df = pd.read_json(row_list, orient='values')
    # print(len(row_byte))
    # print(df.memory_usage(deep=True))
    # print(df)

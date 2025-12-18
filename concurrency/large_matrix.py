import numpy as np
from concurrent.futures import ThreadPoolExecutor
from util import async_timed
import asyncio
from functools import partial


def mean_for_row(arr, row):
    return np.mean(arr[row, :])

data_points = 4_000_000_000
rows = 50
columns = int(data_points / rows)
matrix = np.random.rand(rows, columns)

@async_timed()
async def main():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=16) as pool:
        tasks = [
            loop.run_in_executor(pool, partial(mean_for_row, matrix, row))
            for row in range(rows)
        ]

        results = await asyncio.gather(*tasks)


asyncio.run(main())

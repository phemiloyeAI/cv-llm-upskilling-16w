import time
import asyncio
import requests
from functools import partial
from util import async_timed
from concurrent.futures import ThreadPoolExecutor

def get_status_code(url: str) -> int:
    response = requests.get(url)
    return response.status_code

# start_time = time.time()

# with ThreadPoolExecutor(max_workers=32) as pool:
#     urls = ['https://www.example.com' for _ in range(1000)]
#     results = pool.map(get_status_code, urls)

#     for result in results:
#         print(result)

# end_time = time.time()
# print(f'finished requests in {end_time - start_time:.4f} second(s)')

@async_timed()
async def main():
    loop = asyncio.get_event_loop()

    tasks = []
    with ThreadPoolExecutor(max_workers=1000) as pool:
        urls = ['https://www.example.com' for _ in range(1000)]
        for url in urls:
            tasks.append(loop.run_in_executor(pool, partial(get_status_code, url)))
        
        results = await asyncio.gather(*tasks)
        print(results[0])

asyncio.run(main())
import asyncio
from util import async_timed, delay

# @async_timed()
async def cpu_bound_task():
    counter = 0
    for _ in range(100_000_000):
        counter += 1
    return counter

# @async_timed()
async def main():
    task_1 = asyncio.create_task(cpu_bound_task())
    task_2 = asyncio.create_task(cpu_bound_task())
    delay_task = asyncio.create_task(delay(4))

    await task_1
    await task_2
    await delay_task


asyncio.run(main(), debug=True)
import asyncio
from asyncio import CancelledError

from util import delay

async def n_sequence_sum(n):
    j = 0 
    for i in range(1, n+1):
        j += i
    return j 


async def hello_every_second():
    for i in range(2):
        await asyncio.sleep(1)
        print("I'm running other code while I'm waiting!")

async def main():
    # sleep_for_three = asyncio.create_task(delay(3))
    # sleep_again = asyncio.create_task(delay(3))

    # await hello_every_second()

    # await sleep_for_three
    # await sleep_again

    long_task = asyncio.create_task(delay(10))

    seconds_elapsed = 0

    while not long_task.done():
        print('Task not finished, checking again in a second.')
        await asyncio.sleep(1)
        seconds_elapsed += 1 

        if seconds_elapsed == 5:
            long_task.cancel()

    try:
        await long_task
    except CancelledError:
        print('Our task was cancelled')


async def wait_for_main():

    delay_task = asyncio.create_task(delay(2))

    try:
        result = await asyncio.wait_for(delay_task, timeout=1)
        print(result)

    except asyncio.exceptions.TimeoutError:
        print('Got a timeout')
        print(f'Was the task cancelled? {delay_task.cancelled()}')


async def timeout_shield_main():
    task = asyncio.create_task(delay(10))

    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=5)
        print(result)

    except TimeoutError:
        print("Task took longer than five seconds, it will finish soon!")
        result = await task
        print(result)


asyncio.run(timeout_shield_main())
import time 
from concurrent.futures import ProcessPoolExecutor

def count(count_to: int) -> int:
    start = time.time()
    counter = 0

    while counter < count_to:
        counter = counter + 1
        end = time.time()
    print(f'Finished counting to {count_to} in {end - start}')
    return counter

if __name__ == '__main__':
    with ProcessPoolExecutor() as pool_processor:
        numbers = [100_000_000, 1, 3, 5, 22]
        for result in pool_processor.map(count, numbers):
            print(result)
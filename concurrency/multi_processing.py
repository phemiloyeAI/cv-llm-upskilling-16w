import multiprocessing as mp
import time

def count(count_to):
    start = time.time()
    counter = 0
    while counter  < count_to:
        counter += 1
    end = time.time()
    print(f'Finished counting to {count_to} in {end-start}')
    return counter

def say_hello(name):
    return f'Hi there, {name}'

if __name__ == '__main__':
    # start_time = time.time()

    # to_100m = mp.Process(target=count, args=(200_000_000,))
    # to_200m = mp.Process(target=count, args=(200_000_000,))

    # to_100m.start()
    # to_200m.start()

    # to_100m.join()
    # to_200m.join()

    # end_time = time.time()
    # print(f'Completed in {end_time-start_time}')

    with mp.Pool() as process_pool:
        hi_jeff = process_pool.apply_async(say_hello, args=('Jeff',))
        hi_john = process_pool.apply_async(say_hello, args=('John',))
        print(hi_jeff.get())
        print(hi_john.get())
        print(mp.cpu_count())
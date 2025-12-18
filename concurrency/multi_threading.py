import threading
import os 

def hello_from_process():
    print(f'Hello from child process {os.getpid()}!')

if __name__ == '__main__':
    # hello_process = multiprocessing.Process(target=hello_from_process)

    # hello_process.start()
    # print(f'Hello from parent process {os.getpid()}')
    # hello_process.join()

    # GIL: a python process can only have one thread running python byte code at a time 
    # no two threads can modify 
    import time
    def print_fib(n):
        def fib(n):
            if n == 1: return 0
            elif n == 2: return 1
            else: return fib(n - 1) + fib(n - 2)
        print(f'fib({n}) is {fib(n)}')
    

    def fibs_no_threading():
        print_fib(40)
        print_fib(41)

    def fibs_with_threads():
        fibs_40_thr = threading.Thread(target=print_fib, args=(40, ))
        fibs_41_thr = threading.Thread(target=print_fib, args=(41, ))

        fibs_40_thr.start()
        fibs_41_thr.start()

        fibs_40_thr.join()
        fibs_41_thr.join()
    

    start = time.time()
    fibs_no_threading()
    # fibs_with_threads()
    end = time.time()

    print(f'Threads took {end - start:.4f} seconds.')
    # print(f'Completed in {end - start:.4f} seconds.')
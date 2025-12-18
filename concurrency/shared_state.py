from multiprocessing import Process, Value, Array

def increment_value(shared_int):
    with shared_int.get_lock():
        shared_int.value = shared_int.value + 1
 

if __name__ == '__main__':
    for _ in range(10):
        integer = Value('i', 0)
        procs = [
            Process(target=increment_value, args=(integer,)),
            Process(target=increment_value, args=(integer,))
        ]

        [p.start() for p in procs]
        [p.join() for p in procs]

        print(integer.value)
        assert(integer.value == 2)
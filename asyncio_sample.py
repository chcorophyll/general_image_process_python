#!/usr/bin/env python3

# spinner_thread.py

# credits: Adapted from Michele Simionato's
# multiprocessing example in the python-list:
# https://mail.python.org/pipermail/python-list/2009-February/538048.html
"""
References:
https://github.com/fluentpython/example-code/blob/master/18-asyncio-py3.7/spinner_thread.py
https://github.com/fluentpython/example-code/blob/master/18-asyncio-py3.7/spinner_asyncio.py
"""

# BEGIN SPINNER_THREAD
import threading
import itertools
import time


def spin(msg, done):  # <1>
    for char in itertools.cycle('|/-\\'):  # <3>
        status = char + ' ' + msg
        print(status, flush=True, end='\r')
        if done.wait(0.1):  # <5>
            break
    print(' ' * len(status), end='\r')

def slow_function():  # <7>
    # pretend waiting a long time for I/O
    time.sleep(5)  # <8>
    return 42


def supervisor():  # <9>
    done = threading.Event()
    spinner = threading.Thread(target=spin,
                               args=('thinking!', done))
    print('spinner object:', spinner)  # <10>
    spinner.start()  # <11>
    result = slow_function()  # <12>
    done.set()  # <13>
    spinner.join()  # <14>
    return result


def main():
    result = supervisor()  # <15>
    print('Answer:', result)


if __name__ == '__main__':
    main()
    # import time
    # count_down = 5
    # for i in range(count_down, 0, -1):
    #     msg = u"\r系统将在" + str(i) + "秒内自动退出"
    #     print(msg, end="")
    #     time.sleep(1)
    # end_msg = "结束" + " " * (len(msg) - len("结束"))
    # print(u"\r" + end_msg, end="")
# END SPINNER_THREAD


# BEGIN SPINNER_ASYNCIO
import asyncio
import itertools

async def spin(msg):
    for char in itertools.cycle("|/-\\"):
        status = char + ' ' + msg
        print(status, flush=True, end='\r')
        try:
            await asyncio.sleep(.1)
        except asyncio.CancelledError:
            break
    print(' ' * len(status), end='\r')


async def slow_function():
    await asyncio.sleep(3)
    return 42


asyncio def supervisor():
    spinner = asyncio.create_task(spin("thinking!"))
    print("spinner object:", spinner)
    result = await slow_function()
    spinner.cancel()
    return result


def main():
    result = asyncio.run(supervisor())
    print("Answer:", result)


if __name__ == '__main__':
    main()
# END SPINNER_ASYNCIO

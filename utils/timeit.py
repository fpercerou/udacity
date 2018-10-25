import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print('Running %s...' % (method.__name__))
        result = method(*args, **kw)
        te = time.time()
        hour, min, sec = convertsec(te-ts)
        print('%s took %sh%sm%ss.'
              % (method.__name__, hour, min, sec))
        return result
    return timed

def convertsec(sec):
    seconds = sec % 60
    seconds = int(seconds)
    minutes = (sec / 60) % 60
    minutes = int(minutes)
    hours = (sec / (60 * 60)) % 24
    hours = int(hours)
    return (hours, minutes, seconds)

if __name__ == '__main__':

    @timeit
    def sleep(sec):
        time.sleep(sec)

    sleep(5)

    print(convertsec(10000)) # 2 houes 46 minutes and 40 sec
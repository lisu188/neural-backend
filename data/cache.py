CACHE = {}


def invalidate_cache():
    global CACHE
    CACHE = {}


def cached(f):
    def func():
        name = f.__name__ + str(hash(f))
        if name in CACHE:
            return CACHE[name]
        else:
            CACHE[name] = f()
            return func()

    return func

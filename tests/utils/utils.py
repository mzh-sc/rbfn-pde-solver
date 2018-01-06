import inspect
import time
import re


def duration(func, description=None):
    t_org = time.perf_counter()
    res = func()
    t_end = time.perf_counter()
    print('Duration of {} is {:f}'.format(re.search('duration\((.*)\)', inspect.getsourcelines(func)[0][0]).group(1)
                                        if description is None else description,
                                          t_end - t_org))
    return res

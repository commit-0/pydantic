import sys
if sys.version_info >= (3, 10):
    slots_true = {'slots': True}
else:
    slots_true = {}
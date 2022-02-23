import os
from shutil import copy

wd = os.path.dirname(__file__)
src = os.path.join(wd, 'data_pipeliner', 'output')
dst = os.path.join(wd, 'combine', 'model')

for f in os.listdir(src):
    copy(os.path.join(src, f), dst)

print('Done')
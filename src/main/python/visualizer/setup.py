import os
from shutil import copy

run = r'T:/ABM/ABM_FY19/model_runs/ABM2Plus/v1221/2016_1422new_sxu'
wd = os.path.dirname(os.path.realpath(__file__))

#Set up data pipeline settings
settings_file = os.path.join(wd, 'data_pipeliner', 'config', 'settings.yaml')
copy(os.path.join(wd, 'settings.yaml'), os.path.split(settings_file)[0])

f = open(settings_file, 'r')
data = f.read()
f.close()

data = data.replace('[MODEL_RUN]', run)

f = open(settings_file, 'w')
f.write(data)
f.close()

#Set up combine settings
combine_file = os.path.join(wd, 'combine', 'combine.yaml')
copy(os.path.join(wd, 'combine.yaml'), os.path.split(combine_file)[0])

f = open(combine_file, 'r')
data = f.read()
f.close()

data = data.replace('VISUALIZER_PATH', wd)

f = open(combine_file, 'w')
f.write(data)
f.close()
import os
import pandas as pd
import numpy as np
import openmatrix as omx
import threading
import time

class TableReader(threading.Thread):
    '''
    Multi-threaded table reader

    Parameters
    ----------
    tables (dict):
        Dictionary to put tables in
    names (list):
        List of table names
    fps (list):
        List of filepaths
    '''
    def __init__(self, table, name, fp):
        threading.Thread.__init__(self)
        self.tables = tables
        self.name = name
        self.fp = fp

    def run(self):
        self.tables[self.name] = pd.read_csv(self.fp)

ABMOutputDir = r'T:\ABM\ABM_FY19\model_runs\ABM2Plus\083121\2016_1422_sxu\output'
geogXWalkDir = os.path.split(ABMOutputDir)[0]
SkimDir = r'T:\ABM\ABM_FY19\model_runs\ABM2Plus\083121\2016_1422_sxu\output'

input_files = {'hh':                    ABMOutputDir + r'\householdData_3.csv', #Replace 3 with MAX_ITER (once I find out how MAX_ITER is defined...)
               'per':                   ABMOutputDir + r'\personData_3.csv',
               'tours':                 ABMOutputDir + r'\indivTourData_3.csv',
               'trips':                 ABMOutputDir + r'\indivTripData_3.csv',
               'jtrips':                ABMOutputDir + r'\jointTripData_3.csv',
               'unique_joint_tours':    ABMOutputDir + r'\jointTourData_3.csv',
               'wsLoc':                 ABMOutputDir + r'\wsLocResults_3.csv',
               'aoResults':             ABMOutputDir + r'\aoResults.csv',
               'aoResults_Pre':         ABMOutputDir + r'\aoResults_Pre.csv',
               'visitor_trips':         ABMOutputDir + r'\visitorTrips.csv',
               'mazCorrespondence':     geogXWalkDir + r'\visualizer\data\geographicXwalk_PMSA.csv'
               }
skim_file = SkimDir + r'\traffic_skims_MD.omx'

t0 = time.time()
print('Reading Outputs')
tables = {}
readers = []
for name in input_files:
    readers.append(TableReader(tables, name, input_files[name]))
    readers[-1].start()

for i in range(len(readers)):
    readers[i].join()

t1 = time.time()
print(t1 - t0)

print('Reading Skims')
skim = omx.open_file(skim_file)
zone_mapping = pd.Series(skim.mapping('zone_number')).sort_values()
DST_SKM = pd.DataFrame(skim['MD_SOV_TR_H_DIST'], zone_mapping.index, zone_mapping.index)

t2 = time.time()
print(t2 - t1)

# Prepare files for computing summary statistics
print('Preparing')
tables['aoResults']['HHVEH'] = np.minimum(tables['aoResults']['AO'], 4)
tables['aoResults_Pre']['HHVEH'] = np.minimum(tables['aoResults_Pre']['AO'], 4)
tables['hh']['HHVEH'] = np.minimum(tables['hh']['autos'], 4)

tables['hh']['VEH_NEWCAT'] = np.empty_like(tables['hh'].index)
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 0), 1,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 1) & (tables['hh']['AVs'] == 0), 2,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 1), 3,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 2) & (tables['hh']['AVs'] == 0), 4,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 2), 5,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 1) & (tables['hh']['AVs'] == 1), 6,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 3) & (tables['hh']['AVs'] == 0), 7,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 0) & (tables['hh']['AVs'] == 3), 8,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 2) & (tables['hh']['AVs'] == 1), 9,  tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 1) & (tables['hh']['AVs'] == 2), 10, tables['hh']['VEH_NEWCAT'])
tables['hh']['VEH_NEWCAT'] = np.where((tables['hh']['HVs'] == 4) & (tables['hh']['AVs'] == 0), 11, tables['hh']['VEH_NEWCAT'])

tables['hh']['HHSIZ'] = tables['hh']['hh_id'].map(tables['per'][['hh_id', 'person_id']].groupby('hh_id').count()['person_id'])
tables['hh']['HHSIZE'] = np.minimum(tables['hh']['HHSIZ'], 5)
tables['hh']['ADULTS'] = tables['hh']['hh_id'].map(tables['per'][['hh_id', 'person_id', 'age']].query('age >= 18 and age < 99').groupby('hh_id').count()['person_id'])

tables['per']['PERTYPE'] = tables['per']['type'].map({'Full-time worker': 1,
                                                      'Part-time worker': 2,
                                                      'University student': 3,
                                                      'Non-worker': 4,
                                                      'Retired': 5,
                                                      'Student of driving age': 6,
                                                      'Student of non-driving age': 7,
                                                      'Child too young for school': 8})

tables['mazCorrespondence'] = tables['mazCorrespondence'].set_index('mgra')
tables['wsLoc']['HDISTRICT'] = tables['wsLoc']['HomeMGRA'].map(tables['mazCorrespondence']['pmsa'])
tables['wsLoc']['WDISTRICT'] = tables['wsLoc']['WorkLocation'].map(tables['mazCorrespondence']['pmsa'])
tables['wsLoc']['HHTAZ'] = tables['wsLoc']['HomeMGRA'].map(tables['mazCorrespondence']['taz'])
tables['wsLoc']['WTAZ'] = tables['wsLoc']['WorkLocation'].map(tables['mazCorrespondence']['taz']).fillna(0).astype(int)
tables['wsLoc']['STAZ'] = tables['wsLoc']['SchoolLocation'].map(tables['mazCorrespondence']['taz']).fillna(0).astype(int)

tables['wsLoc']['HHWTAZ'] = list(zip(tables['wsLoc']['HHTAZ'], tables['wsLoc']['WTAZ']))
tables['wsLoc']['HHSTAZ'] = list(zip(tables['wsLoc']['HHTAZ'], tables['wsLoc']['STAZ']))

tables['wsLoc']['WorkLocationDistance'] = np.where(tables['wsLoc']['WTAZ'] > 0, tables['wsLoc']['HHWTAZ'].apply(lambda flow: DST_SKM.loc[flow[0], flow[1]]), 0)
tables['wsLoc']['SchoolLocationDistance'] = np.where(tables['wsLoc']['STAZ'] > 0, tables['wsLoc']['HHSTAZ'].apply(lambda flow: DST_SKM.loc[flow[0], flow[1]]), 0)

t3 = time.time()
print(t3 - t2)

print('Done')
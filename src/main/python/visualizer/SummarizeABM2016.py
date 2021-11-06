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

def pivot(df, row, col, val = None, f = sum):
    if val is None:
        return df[[row, col]].reset_index().groupby([row, col]).count()['index'].reset_index().pivot(row, col, 'index').fillna(0)
    else:
        return df[[row, col, val]].groupby([row, col]).f()[val].reset_index().pivot(row, col, val).fillna(0)

ABMOutputDir = r'C:\test\visualizer_conversion\output' #Copied from T:\ABM\ABM_FY19\model_runs\ABM2Plus\v1221\2016_1422new_sxu\output
geogXWalkDir = os.path.split(ABMOutputDir)[0]
SkimDir = r'T:\ABM\ABM_FY19\model_runs\ABM2Plus\v1221\2016_1422new_sxu\output'
vizOutputDir = r'C:\test\visualizer_conversion\visualizer\outputs\summaries\BUILD'#os.path.split(ABMOutputDir)[0] + r'\visualizer\outputs\summaries\BUILD'

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

purpose_map = {'Work': 1,
               'University': 2,
               'School': 3,
               'Escort': 4,
               'Shop': 5,
               'Maintenance': 6,
               'Eating Out': 7,
               'Visiting': 8,
               'Discretionary': 9,
               'Work-Based': 10,
               'work related': 10}

t0 = time.time()
print('Reading ABM Outputs')
tables = {}
readers = []
for name in input_files:
    readers.append(TableReader(tables, name, input_files[name]))
    readers[-1].start()

for i in range(len(readers)):
    readers[i].join()

#Update individual tour and trip files so that tours have unique IDs
maxtours = tables['tours']['person_id'].value_counts().max() # Maximum number of tours a person makes in a day
#tables['tours'] = tables['tours'].sort_values('person_id') # Might not be necessary, but just to be safe...
#tables['tours']['match_back0'] = np.ones_like(tables['tours'].index)
#for i in range(1, maxtours):
#    tables['tours']['pid_back%d'%(i)] = tables['tours']['person_id']
#    tables['tours']['pid_back%d'%(i)] = np.hstack((i*[0], tables['tours']['person_id'].iloc[:-i]))
#    tables['tours']['match_back%d'%(i)] = (tables['tours']['person_id'] == tables['tours']['pid_back%d'%(i)]).astype(int)
#tables['tours']['tour_id2'] = tables['tours'][['match_back%d'%(i) for i in range(maxtours)]].sum(1)
tables['tours']['cummulative_tours'] = range(tables['tours'].shape[0])
tables['tours']['last_person'] = np.hstack(([0], tables['tours']['person_id'][:-1]))
tables['tours']['new_person'] = (tables['tours']['person_id']) != (tables['tours']['last_person'])
tables['tours']['to_subtract'] = tables['tours']['new_person'] * np.hstack(([0], tables['tours']['cummulative_tours'].iloc[:-1]))
for i in range(maxtours):
    tables['tours']['to_subtract'] = np.where(tables['tours']['to_subtract'] == 0,
                                              np.hstack(([0], tables['tours']['to_subtract'].iloc[:-1])),
                                              tables['tours']['to_subtract'])
tables['tours']['tour_id2'] = tables['tours']['cummulative_tours'] - tables['tours']['to_subtract']

maxtrips = tables['trips']['person_id'].value_counts().max() #Maximum number of trips a person makes in a day
tables['trips']['new_tour'] = (tables['trips']['orig_purpose'] == 'Home') | ((tables['trips']['tour_purpose'] == 'Work-Based') & (tables['trips']['orig_purpose'] == 'Work'))
tables['trips']['cummulative_tours'] = np.cumsum(tables['trips']['new_tour'])
tables['trips']['last_person'] = np.hstack(([0], tables['trips']['person_id'][:-1]))
tables['trips']['new_person'] = (tables['trips']['person_id']) != (tables['trips']['last_person'])
tables['trips']['to_subtract'] = tables['trips']['new_person'] * np.hstack(([0], tables['trips']['cummulative_tours'].iloc[:-1]))
for i in range(maxtrips):
    tables['trips']['to_subtract'] = np.where(tables['trips']['to_subtract'] == 0,
                                              np.hstack(([0], tables['trips']['to_subtract'].iloc[:-1])),
                                              tables['trips']['to_subtract'])
tables['trips']['tour_id2'] = tables['trips']['cummulative_tours'] - tables['trips']['to_subtract']
#tables['trips']['end_wb_tour'] = (tables['trips']['tour_purpose'] == 'Work-Based') | (tables['trips']['dest_purpose'] == 'Work')

t1 = time.time()
print(t1 - t0)

print('Reading Skims')
skim = omx.open_file(skim_file)
skimLookup = pd.Series(skim.mapping('zone_number')).sort_values()
N = len(skimLookup)
#DST_SKM = np.concatenate((np.reshape(skim['MD_SOV_TR_H_DIST'], N**2), [-1]))
DST_SKM = np.reshape(skim['MD_SOV_TR_H_DIST'], N**2)

t2 = time.time()
print(t2 - t1)

# Prepare files for computing summary statistics
print('Preparing Data')
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

tables['wsLoc']['HHIndex'] = tables['wsLoc']['HHTAZ'].map(skimLookup).fillna(0).astype(int)
tables['wsLoc']['WIndex'] = tables['wsLoc']['WTAZ'].map(skimLookup).fillna(0).astype(int)
tables['wsLoc']['SIndex'] = tables['wsLoc']['STAZ'].map(skimLookup).fillna(0).astype(int)
#tables['wsLoc']['HHWIndex'] = np.where(tables['wsLoc']['WTAZ'] > 0, N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['WIndex'], -1)
#tables['wsLoc']['HHSIndex'] = np.where(tables['wsLoc']['STAZ'] > 0, N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['SIndex'], -1)

#tables['wsLoc']['WorkLocationDistance'] = DST_SKM[tables['wsLoc']['HHWIndex']]#.apply(lambda i: DST_SKM[i])
#tables['wsLoc']['SchoolLocationDistance'] = DST_SKM[tables['wsLoc']['HHSIndex']]#.apply(lambda i: DST_SKM[i])

tables['wsLoc']['WorkLocationDistance'] = DST_SKM[N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['WIndex']]
tables['wsLoc']['SchoolLocationDistance'] = DST_SKM[N*tables['wsLoc']['HHIndex'] + tables['wsLoc']['SIndex']]

t3 = time.time()
print(t3 - t2)

print('Computing Summary Statistics')
tables['aoResults_Pre'][['HHVEH', 'HHID']].groupby('HHVEH').count()['HHID'].reset_index().to_csv(vizOutputDir + r'\autoOwnership_Pre.csv')
tables['aoResults'][['HHVEH', 'HHID']].groupby('HHVEH').count()['HHID'].reset_index().to_csv(vizOutputDir + r'\autoOwnership.csv')
tables['hh'][['AVs', 'hh_id']].groupby('AVs').count()['hh_id'].reset_index().to_csv(vizOutputDir + r'\autoOwnership_AV.csv')
tables['hh'][['VEH_NEWCAT', 'hh_id']].groupby('VEH_NEWCAT').count()['hh_id'].reset_index().to_csv(vizOutputDir + r'\autoOwnership_new.csv')

# Zero auto HHs by TAZ
tables['hh']['HHTAZ'] = tables['hh']['home_mgra'].map(tables['mazCorrespondence']['taz'])
tables['hh']['ZeroAutoWGT'] = np.where(tables['hh']['HHVEH'] == 0, 1, 0)
tables['hh']['ZeroAutoWGT'] = tables['hh']['ZeroAutoWGT'].fillna(0)
tables['hh'][['HHTAZ', 'ZeroAutoWGT']].groupby('HHTAZ').sum()['ZeroAutoWGT'].reset_index().to_csv(vizOutputDir + r'\zeroAutoByTaz.csv')

tables['per'][['PERTYPE', 'person_id']].groupby('PERTYPE').count()['person_id'].reset_index().to_csv(vizOutputDir + r'\pertypeDistbn.csv')
tables['per'][['tele_choice', 'person_id']].groupby('tele_choice').count()['person_id'].reset_index().to_csv(vizOutputDir + r'\teleCommute_frequency.csv')
tables['hh'][['transponder', 'hh_id']].groupby('transponder').count()['hh_id'].reset_index().to_csv(vizOutputDir + r'\transponder_ownership.csv')

# Micro-mobility
micro_r1 = tables['trips']['micro_walkMode'].reset_index().groupby('micro_walkMode').count()['index']
micro_r2 = tables['trips']['micro_trnAcc'].reset_index().groupby('micro_trnAcc').count()['index']
micro_r3 = tables['trips']['micro_trnEgr'].reset_index().groupby('micro_trnEgr').count()['index']

micro_v1 = tables['visitor_trips']['micro_walkMode'].reset_index().groupby('micro_walkMode').count()['index']
micro_v2 = tables['visitor_trips']['micro_trnAcc'].reset_index().groupby('micro_trnAcc').count()['index']
micro_v3 = tables['visitor_trips']['micro_trnEgr'].reset_index().groupby('micro_trnEgr').count()['index']

micromobility_summary = micro_r1 + micro_r2 + micro_r3 + micro_v1 + micro_v2 + micro_v3
micromobility_summary.index.name = 'micro_mode'
pd.DataFrame({'trips': micromobility_summary}).reset_index().to_csv(vizOutputDir + r'\micormobility.csv')

# Mandatory DC
workers = tables['wsLoc'].query('WorkLocation > 0 and WorkLocation != 99999').dropna(subset = ['WorkLocationDistance'])
students = tables['wsLoc'].query('SchoolLocation > 0 and SchoolLocation != 88888').dropna(subset = ['SchoolLocationDistance'])

# Code distance bins
workers['distbin'] = np.where(workers['WorkLocationDistance'] % 1 == 0, workers['WorkLocationDistance'] + 1, np.minimum(np.ceil(workers['WorkLocationDistance']), 51)).astype(int)
students['distbin'] = np.where(students['SchoolLocationDistance'] % 1 == 0, students['SchoolLocationDistance'] + 1, np.minimum(np.ceil(students['SchoolLocationDistance']), 51)).astype(int)

# Create subsets for university and school students
univ = students.query('PersonType == 3')
schl = students.query('PersonType >= 6')

# Compute TLFDs by district and total
tlfd_work = workers.reset_index()[['distbin', 'HDISTRICT', 'index']].groupby(['distbin', 'HDISTRICT']).count()['index'].reset_index().pivot('distbin', 'HDISTRICT', 'index').fillna(0)
tlfd_univ = univ.reset_index()[['distbin', 'HDISTRICT', 'index']].groupby(['distbin', 'HDISTRICT']).count()['index'].reset_index().pivot('distbin', 'HDISTRICT', 'index').fillna(0)
tlfd_schl = schl.query('PersonType >= 6').reset_index()[['distbin', 'HDISTRICT', 'index']].groupby(['distbin', 'HDISTRICT']).count()['index'].reset_index().pivot('distbin', 'HDISTRICT', 'index').fillna(0)

for tlfd in [tlfd_work, tlfd_univ, tlfd_schl]:
    for i in tlfd.columns:
        tlfd['District_{}'.format(i)] = tlfd[i]
        del tlfd[i]
    tlfd['Total'] = tlfd.sum(1)

tlfd_work.to_csv(vizOutputDir + r'\workTLFD.csv')
tlfd_univ.to_csv(vizOutputDir + r'\univTLFD.csv')
tlfd_schl.to_csv(vizOutputDir + r'\schlTLFD.csv')

# Output avg trip lengths for visualizer
workTripLengths = workers[['HDISTRICT', 'WorkLocationDistance']].groupby('HDISTRICT').sum()['WorkLocationDistance'] / workers[['HDISTRICT', 'WorkLocationDistance']].groupby('HDISTRICT').count()['WorkLocationDistance']
workTripLengths.index.name = 'District'
workTripLengths.loc['Total'] = workers['WorkLocationDistance'].sum() / workers.shape[0]

univTripLengths = univ[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').sum()['SchoolLocationDistance'] / univ[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').count()['SchoolLocationDistance']
univTripLengths.loc['Total'] = univ['SchoolLocationDistance'].sum() / univ.shape[0]

schlTripLengths = schl[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').sum()['SchoolLocationDistance'] / schl[['HDISTRICT', 'SchoolLocationDistance']].groupby('HDISTRICT').count()['SchoolLocationDistance']
schlTripLengths.loc['Total'] = schl['SchoolLocationDistance'].sum() / schl.shape[0]

pd.DataFrame({'Work': workTripLengths, 'Univ': univTripLengths, 'Schl': schlTripLengths}).to_csv(vizOutputDir + r'\mandTripLengths.csv')

# Work from home [for each district and total]
districtWorkers = tables['wsLoc'][['WorkLocation', 'HDISTRICT']].query('WorkLocation > 0').groupby('HDISTRICT').count()['WorkLocation']
districtWfh = tables['wsLoc'][['WorkLocation', 'HDISTRICT']].query('WorkLocation == 99999').groupby('HDISTRICT').count()['WorkLocation']
wfh_summary = pd.DataFrame({'Workers': districtWorkers, 'WFH': districtWfh})
wfh_summary.index.name = 'District'
wfh_summary.loc['Total'] = wfh_summary.sum(0)
wfh_summary.to_csv(vizOutputDir + r'\wfh_summary.csv')
totalwfh = pd.DataFrame(wfh_summary.loc['Total']).T
totalwfh.index.name = 'District'
totalwfh.to_csv(vizOutputDir + r'\wfh_summary_region.csv')

#County-County Flows
#JJF: Don't we only have one county? :)
countyFlows = workers[['HDISTRICT', 'WDISTRICT', 'PersonID']].groupby(['HDISTRICT', 'WDISTRICT']).count()['PersonID'].reset_index().pivot('HDISTRICT', 'WDISTRICT', 'PersonID').fillna(0)
countyFlows.loc['Total'] = countyFlows.sum(0)
countyFlows['Total'] = countyFlows.sum(1)
districts = countyFlows.columns
for i in range(2): #Rename the columns, then flip and rename the rows, then flip again
    for col in districts:
        try:
            countyFlows['District_{}'.format(int(col))] = countyFlows[col]            
        except ValueError:
            countyFlows['District_' + col] = countyFlows[col]
        del countyFlows[col]
    countyFlows = countyFlows.T
countyFlows.index.name = None
countyFlows.columns.name = None
countyFlows.to_csv(vizOutputDir + r'\countyFlows.csv')

t4 = time.time()
print(t4 - t3)

print('Processing Tour Files')
tables['tours']['PERTYPE'] = tables['tours']['person_type']
tables['tours']['DISTMILE'] = tables['tours']['tour_distance']
tables['tours']['HHVEH'] = tables['tours']['hh_id'].map(tables['hh'].set_index('hh_id')['HHVEH'])
tables['tours']['ADULTS'] = tables['tours']['hh_id'].map(tables['hh'].set_index('hh_id')['ADULTS'])
tables['tours']['AUTOSUFF'] = np.where(tables['tours']['HHVEH'] == 0, 0,
                                       np.where(tables['tours']['HHVEH'] < tables['tours']['ADULTS'], 1, 2))

tables['tours']['num_tot_stops'] = tables['tours']['num_ob_stops'] + tables['tours']['num_ib_stops']

tables['tours']['OTAZ'] = tables['tours']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['tours']['DTAZ'] = tables['tours']['dest_mgra'].map(tables['mazCorrespondence']['taz'])

tables['tours']['oindex'] = tables['tours']['OTAZ'].map(skimLookup)
tables['tours']['dindex'] = tables['tours']['DTAZ'].map(skimLookup)
tables['tours']['odindex'] = N*tables['tours']['oindex'] + tables['tours']['dindex']
tables['tours']['SKIMDIST'] = DST_SKM[N*tables['tours']['oindex'] + tables['tours']['dindex']]

tables['unique_joint_tours']['HHVEH'] = tables['unique_joint_tours']['hh_id'].map(tables['hh'].set_index('hh_id')['HHVEH'])
tables['unique_joint_tours']['ADULTS'] = tables['unique_joint_tours']['hh_id'].map(tables['hh'].set_index('hh_id')['ADULTS'])
tables['unique_joint_tours']['AUTOSUFF'] = np.where(tables['unique_joint_tours']['HHVEH'] == 0, 0,
                                                    np.where(tables['unique_joint_tours']['HHVEH'] < tables['unique_joint_tours']['ADULTS'], 1, 2))

#Code tour purposes
tables['tours']['TOURPURP'] = tables['tours']['tour_purpose'].map(purpose_map)
tables['tours']['TOURCAT'] = np.where(tables['tours']['TOURPURP'] <= 3, 0,
                                      np.where(tables['tours']['TOURPURP'] <= 9, 1, 2))

#Compute duration
tables['tours']['tourdur'] = tables['tours']['end_period'] - tables['tours']['start_period'] + 1 # to match survey
tables['tours']['TOURMODE'] = tables['tours']['tour_mode']

# exclude school escorting stop from ride sharing mandatory tours
tables['unique_joint_tours']['JOINT_PURP'] = tables['unique_joint_tours']['tour_purpose'].map(purpose_map)
tables['unique_joint_tours']['NUMBER_HH'] = (tables['unique_joint_tours']['tour_participants'].apply(len) + 1) // 2 # Number of people on tour

# get participant IDs and person types for each participant
#tables['per']['hhper_id'] = list(zip(tables['per']['hh_id'], tables['per']['person_num']))
tables['per']['hhper_id'] = tables['per']['hh_id'].astype(str) + '-' + tables['per']['person_num'].astype(str)
for i in range(1, 9):
    def get_perno(str):
        try:
            return str.replace(' ', '')[i-1]
        except IndexError:
            return 0
    tables['unique_joint_tours']['PER%d'%(i)] = tables['unique_joint_tours']['tour_participants'].apply(get_perno)
    tables['unique_joint_tours']['PTYPE%d'%(i)] = (tables['unique_joint_tours']['hh_id'].astype(str) + '-' + tables['unique_joint_tours']['PER%d'%(i)].astype(str)).map(tables['per'].set_index('hhper_id')['PERTYPE']).fillna(0)

tables['unique_joint_tours']['num_tot_stops'] = tables['unique_joint_tours']['num_ob_stops'] + tables['unique_joint_tours']['num_ib_stops']
tables['unique_joint_tours']['OTAZ'] = tables['unique_joint_tours']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['unique_joint_tours']['DTAZ'] = tables['unique_joint_tours']['dest_mgra'].map(tables['mazCorrespondence']['taz'])
tables['unique_joint_tours']['tourdur'] = tables['unique_joint_tours']['end_period'] - tables['unique_joint_tours']['start_period'] + 1
tables['unique_joint_tours']['TOURMODE'] = tables['unique_joint_tours']['tour_mode']

# ----
# this part is added by nagendra.dhakar@rsginc.com from binny.paul@rsginc.com soabm summaries
# translated from R to Python by joe.flood@sandag.org

# create a combined temp tour file for creating stop freq model summary
temp_tour1 = tables['tours'][['TOURPURP', 'num_ob_stops', 'num_ib_stops']]
temp_tour2 = tables['unique_joint_tours'][['JOINT_PURP', 'num_ob_stops', 'num_ib_stops']]
temp_tour2['TOURPURP'] = temp_tour2['JOINT_PURP']
del temp_tour2['JOINT_PURP']
temp_tour = pd.concat((temp_tour1, temp_tour2))

# code stop frequency model alternatives
temp_tour['STOP_FREQ_ALT'] = 4*np.minimum(temp_tour['num_ob_stops'], 3) + np.minimum(temp_tour['num_ib_stops'], 3) + 1
stopFreqModel_summary = pivot(temp_tour, 'STOP_FREQ_ALT', 'TOURPURP')
stopFreqModel_summary.index.name = None
stopFreqModel_summary.columns.name = None
stopFreqModel_summary.to_csv(vizOutputDir + r'\stopFreqModel_summary.csv')

# ------

t5 = time.time()
print(t5 - t4)

print('Processing trip file')
tables['trips']['TOURMODE'] = tables['trips']['tour_mode']
tables['trips']['TRIPMODE'] = tables['trips']['trip_mode']
tables['trips']['TOURPURP'] = tables['trips']['tour_purpose'].map(purpose_map)
tables['trips']['OPURP'] = tables['trips']['orig_purpose'].map(purpose_map)
tables['trips']['DPURP'] = tables['trips']['dest_purpose'].map(purpose_map)
tables['trips']['TOURCAT'] = np.where(tables['trips']['TOURPURP'] <= 3, 0,
                                      np.where(tables['trips']['TOURPURP'] <= 9, 1, 2))

#Mark stops and get other attributes
nr = tables['trips'].shape[0]
tables['trips']['inb_next'] = np.zeros_like(tables['trips'].index)
tables['trips']['inb_next'].iloc[:nr-1] = tables['trips']['inbound'][1:]
tables['trips']['stops'] = np.where((tables['trips']['DPURP'] > 0) & (((tables['trips']['inbound'] == 0) & (tables['trips']['inb_next'] == 0)) | ((tables['trips']['inbound'] == 1) & (tables['trips']['inb_next'] == 1))),
                                    1, 0)

tables['trips']['OTAZ'] = tables['trips']['orig_mgra'].map(tables['mazCorrespondence']['taz'])
tables['trips']['DTAZ'] = tables['trips']['dest_mgra'].map(tables['mazCorrespondence']['taz'])

cols = ['hh_id', 'person_num', 'TOURCAT', 'tour_id2']
coefs = [100000, 1000, 100, 1]
tables['tours']['lookup'] = tables['tours'][cols].dot(coefs)
tables['trips']['TOUROTAZ'] = tables['trips'][cols].dot(coefs).map(tables['tours'].set_index('lookup')['OTAZ'])
tables['trips']['TOURDTAZ'] = tables['trips'][cols].dot(coefs).map(tables['tours'].set_index('lookup')['DTAZ'])

tables['trips']['oindex'] = tables['trips']['OTAZ'].map(skimLookup)
tables['trips']['dindex'] = tables['trips']['DTAZ'].map(skimLookup)
tables['trips']['od_dist'] = DST_SKM[N*tables['trips']['oindex'] + tables['trips']['dindex']]

#create stops table
stops = tables['trips'].query('stops == 1')
stops['finaldestTAZ'] = np.where(stops['inbound'] == 0, stops['TOURDTAZ'], stops['TOUROTAZ'])

stops['oindex'] = stops['OTAZ'].map(skimLookup)
stops['dindex'] = stops['finaldestTAZ'].map(skimLookup)
stops['od_dist'] = DST_SKM[N*stops['oindex'] + stops['dindex']]

stops['oindex2'] = stops['OTAZ'].map(skimLookup)
stops['dindex2'] = stops['DTAZ'].map(skimLookup)
stops['os_dist'] = DST_SKM[N*stops['oindex2'] + stops['dindex2']]

stops['oindex3'] = stops['DTAZ'].map(skimLookup)
stops['dindex3'] = stops['finaldestTAZ'].map(skimLookup)
stops['sd_dist'] = DST_SKM[N*stops['oindex3'] + stops['dindex3']]

stops['out_dir_dist'] = stops['os_dist'] + stops['sd_dist'] - stops['od_dist']

t6 = time.time()
print(t6 - t5)

print('Done')
print(t6 - t0)
import os
import simpy
import random
import math
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################
###################################################################################################
####MODEL
###################################################################################################
###################################################################################################
###################################################################################################

class default_params():
    #########Set Up
    run_name = 'Limited Beds'
    run_time = 24*60*(365 + 20) #One year plus 20 day warm up time
    run_days = int(run_time/(60*24)) 
    iterations = 3#0
    sample_time = 1
    print_outs = False
    occ_sample_time = 60
    pat_res = []
    occ_res = []

    #########Location Capacities
    inf_model = False

    if inf_model:
        ED_beds = MRU_beds = SAU_beds = SDEC_beds = FSDEC_beds = SW_beds = np.inf
    else:
        ED_beds = 105
        MRU_beds = 47
        SAU_beds = 19
        SDEC_beds = 45
        FSDEC_beds = 10
        SW_beds = 850

    #########Read in the data
    conn_str = open('G:/PerfInfo/Performance Management/OR Team/Useful Files/uhpbi conn string.txt', 'r')
    uhpbi_engine = create_engine(conn_str.read())
    conn_str.close()
    #Get query and read in the data
    SQL = open('G:/PerfInfo/Performance Management/OR Team/Emily Projects/Discrete Event Simulation/Medical Take/SQL/FullHospitalModel.sql', 'r')
    df = pd.read_sql_query(SQL.read(), uhpbi_engine)
    SQL.close()
    uhpbi_engine.dispose()

    #########timings
    ED_LoS = 4*60
    MRU_LoS = 12*60
    SAU_LoS = 12*60
    SDEC_FSDEC_LoS = 6*60
    Spec_LoS = df.groupby('PerformanceSpecialty')['Ward_LoS'].mean()

    #########Pathways
    #Get the probabilities of each pathway in the data
    def build_path(row):
        path = ['start', row['Arrival']]
        for col in ["MRU or SAU", "SDEC", "Spec Ward"]:
            if pd.notna(row[col]) and row[col] != "":
                path.append(row[col])
        path.append("Home")
        return path

    paths = df.apply(build_path, axis=1)
    transitions = pd.Series([(p[i], p[i+1]) for p in paths for i in range(len(p)-1)]
                            ).value_counts().reset_index()
    transitions.columns = ["pair", "count"]
    transitions[["from", "to"]] = pd.DataFrame(transitions["pair"].tolist())
    transitions["prob"] = transitions["count"] / transitions.groupby("from")["count"].transform("sum")

    # Route lookup disctionary
    routing = defaultdict(lambda: ([], []))
    for _, r in transitions.iterrows():
        routing[r["from"]][0].append(r["to"])
        routing[r["from"]][1].append(r["prob"])

    #Specialty probabilities
    spec = df['PerformanceSpecialty'].value_counts()
    spec = spec / spec.sum()

    #######Arrivals
    df['arrival'] = (df['ED_arr'].fillna(df['SDEC_arr']).fillna(df['MRU_SAU_arr'])
                     .fillna(df['Ward_arr']))
    arr = df.loc[df['arrival'] >= pd.to_datetime('2026-09-01'), 'arrival'].sort_values()
    inter_arr = (arr.shift(-1) - arr).dt.total_seconds() / 60
    mean_inter_arr = inter_arr.mean()

class spawn_patient:
    def __init__(self, p_id):
        #patient id
        self.id = p_id

        #Generate Patient Pathway
        self.path = []
        self.node = 'start'
        for _ in range(5):
            if self.node == 'end' or self.node not in default_params.routing:
                break
            nexts, probs = default_params.routing[self.node]
            self.node = random.choices(nexts, weights=probs, k=1)[0]
            self.path.append(self.node)
   
        #Arrivals and Discharge times and locations
        self.model_arr = np.nan
        self.ED_queue = np.nan
        self.ED_arr = np.nan
        self.ED_dis = np.nan
        self.MRU_or_SAU = next((i for i in self.path if 'SAU' in i or 'MRU' in i), None)
        self.MoS_queue = np.nan
        self.MoS_arr = np.nan
        self.MoS_dis = np.nan
        self.FSDEC_or_SDEC = next((i for i in self.path if 'FSDEC' in i or 'SDEC' in i), None)
        self.SDEC_queue = np.nan
        self.SDEC_arr = np.nan
        self.SDEC_dis = np.nan
        self.Specialty = (random.choices(default_params.spec.index,
                                              default_params.spec.tolist(), k=1)[0]
                                if 'Specialty Ward' in self.path else np.nan)
        self.Spec_queue = np.nan
        self.Spec_arr = np.nan
        self.Spec_dis = np.nan
        self.model_dis = np.nan

class Medical_Take_Model:
    def __init__(self, run_number, input_params):
        #Set up lists to record results
        self.patient_results = []
        self.occupancy_results = []
        #start environment, set patient counter to 0 and set run number
        self.env = simpy.Environment()
        self.input_params = input_params
        self.patient_counter = 0
        self.run_number = run_number

        #establish location resources
        self.ED = simpy.PriorityResource(self.env, capacity=input_params.ED_beds)
        self.MRU = simpy.PriorityResource(self.env, capacity=input_params.MRU_beds)
        self.SAU = simpy.PriorityResource(self.env, capacity=input_params.SAU_beds)
        self.SDEC = simpy.PriorityResource(self.env, capacity=input_params.SDEC_beds)
        self.FSDEC = simpy.PriorityResource(self.env, capacity=input_params.FSDEC_beds)
        self.SW = simpy.PriorityResource(self.env, capacity=input_params.SW_beds)

    def arrivals(self):
        while True:
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter)
            #print(f'Patient {self.patient_counter} starting {p.path} at {self.env.now}')
            self.env.process(self.Hospital_Journey(p))
            sampled_interarrival = round(random.expovariate(1.0 / self.input_params.mean_inter_arr))
            #print(f'Time out for {sampled_interarrival} until next patient')
            yield self.env.timeout(sampled_interarrival)

    def Hospital_Journey(self, pat):
        pat.model_arr = self.env.now
        for loc in pat.path:
            #print(f'{pat.id}, {loc}, {self.env.now}')

            #ED
            if loc == 'ED':
                pat.ED_queue = self.env.now
                with self.ED.request() as req:
                    yield req
                    pat.ED_arr = self.env.now
                    sample_ED_time = round((random.expovariate(1.0
                                                / self.input_params.ED_LoS)))

                    yield self.env.timeout(sample_ED_time)
                pat.ED_dis = self.env.now

            #MRU
            if loc == 'MRU':
                pat.MoS_queue = self.env.now
                with self.MRU.request() as req:
                    yield req
                    pat.MoS_arr = self.env.now
                    sample_MRU_time = round((random.expovariate(1.0
                                                / self.input_params.MRU_LoS)))
                    yield self.env.timeout(sample_MRU_time)
                pat.MoS_dis = self.env.now

            #SAU
            if loc == 'SAU':
                pat.MoS_queue = self.env.now
                with self.SAU.request() as req:
                    yield req
                    pat.MoS_arr = self.env.now
                    sample_SAU_time = round((random.expovariate(1.0
                                                / self.input_params.SAU_LoS)))
                    yield self.env.timeout(sample_SAU_time)
                pat.MoS_dis = self.env.now

            #SDEC
            if loc == 'SDEC':
                pat.SDEC_queue = self.env.now
                with self.SDEC.request() as req:
                    yield req
                    pat.SDEC_arr = self.env.now
                    sample_SDEC_time = round((random.expovariate(1.0
                                                / self.input_params.SDEC_FSDEC_LoS)))
                    yield self.env.timeout(sample_SDEC_time)
                pat.SDEC_dis = self.env.now

            #FSDEC
            if loc == 'FSDEC':
                pat.SDEC_queue = self.env.now
                with self.FSDEC.request() as req:
                    yield req
                    pat.SDEC_arr = self.env.now
                    sample_FSDEC_time = round((random.expovariate(1.0
                                                / self.input_params.SDEC_FSDEC_LoS)))
                    yield self.env.timeout(sample_FSDEC_time)
                pat.SDEC_dis = self.env.now

            #Specialty Ward
            if loc == 'Specialty Ward':
                pat.Spec_queue = self.env.now
                with self.SW.request() as req:
                    yield req
                    pat.Spec_arr = self.env.now
                    Av_LoS = self.input_params.Spec_LoS.loc[pat.Specialty]
                    sample_Spec_time = round((random.expovariate(1.0 / Av_LoS)))
                    yield self.env.timeout(sample_Spec_time)
                pat.Spec_dis = self.env.now
        

        pat.model_dis = self.env.now
        self.store_patient_results(pat)


    def store_patient_results(self, patient):
        self.patient_results.append([self.run_number,
                                     patient.id,
                                     patient.path,
                                     patient.model_arr,
                                     patient.ED_queue,
                                     patient.ED_arr,
                                     patient.ED_dis,
                                     patient.MRU_or_SAU,
                                     patient.MoS_queue,
                                     patient.MoS_arr,
                                     patient.MoS_dis,
                                     patient.FSDEC_or_SDEC,
                                     patient.SDEC_queue,
                                     patient.SDEC_arr,
                                     patient.SDEC_dis,
                                     patient.Specialty,
                                     patient.Spec_queue,
                                     patient.Spec_arr,
                                     patient.Spec_dis,
                                     patient.model_dis])

    def store_occupancy(self):
        while True:
            self.occupancy_results.append([self.run_number,
                                               self.ED._env.now,
                                               len(self.ED.queue),
                                               self.ED.count,
                                               len(self.MRU.queue),
                                               self.MRU.count,
                                               len(self.SAU.queue),
                                               self.SAU.count,
                                               len(self.SDEC.queue),
                                               self.SDEC.count,
                                               len(self.FSDEC.queue),
                                               self.FSDEC.count,
                                               len(self.SW.queue),
                                               self.SW.count])

            yield self.env.timeout(self.input_params.occ_sample_time)

    def run(self):
        self.env.process(self.arrivals())
        self.env.process(self.store_occupancy())
        self.env.run(until = self.input_params.run_time)
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.occupancy_results
        return self.patient_results, self.occupancy_results

def export_results(pat_results, occ_results):
    pat_df = pd.DataFrame(pat_results,
                          columns = ['Run', 'Pat', 'Path', 'Model Arr', 'ED Queue', 'ED Arr', 'ED Dis', 'MRU or SAU',
                                     'MRU or SAU Queue', 'MRU or SAU Arr', 'MRU or SAU Dis', '(F)SDEC', 'SDEC Queue',
                                     'SDEC Arr', 'SDEC Dis', 'Specialty', 'Spec Queue', 'Spec Arr', 'Spec Dis', 'Home'])

    occ_df = pd.DataFrame(occ_results,
                          columns = ['Run', 'Time', 'ED Queue', 'ED Occ', 'MRU Queue', 'MRU Occ',
                                     'SAU Queue', 'SAU Occ', 'SDEC Queue', 'SDEC Occ', 'FSDEC Queue',
                                     'FSDEC Occ', 'Spec Queue', 'Spec Occ'])
    occ_df['Day'] = (occ_df['Time'] // (24*60)) - 20
    return pat_df, occ_df

def run_the_model(input_params):
    #run the model for the number of iterations specified
    for run in range(input_params.iterations):
        print(f"Run {run+1} of {input_params.iterations}")
        model = Medical_Take_Model(run, input_params)
        model.run()
    patient_df, occ_df = export_results(input_params.pat_res,
                                        input_params.occ_res)
    return patient_df, occ_df

pat, occ = run_the_model(default_params)
#Create results folder if not exist
path = f'G:/PerfInfo/Performance Management/OR Team/Emily Projects/Discrete Event Simulation/Medical Take/Results/{default_params.run_name}'
os.makedirs(path, exist_ok=True)  # create if it doesn't exist (including parent dirs)
os.chdir(path)

for file in ['Full Outputs', 'Plots']:
    os.makedirs(file, exist_ok=True)

#Save full data
pat.to_csv(f'Full Outputs/patients - {default_params.run_name}.csv')
occ.to_csv(f'Full Outputs/occupancy - {default_params.run_name}.csv')


###################################################################################################
###################################################################################################
###################################################################################################
####RESULTS
###################################################################################################
###################################################################################################
###################################################################################################
# 25th and 75th Percentiles functions 
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)

#Remove warm up period
occ = occ.loc[occ['Day'] > 0].copy()

###################################################################################################
####4hr Performance
###################################################################################################
pat['ED Leave'] = np.where(pat['ED Arr'], pat['MRU or SAU Arr'].fillna(pat['SDEC Arr']).fillna(pat['Spec Arr']).fillna(pat['Home']), np.nan)
pat['ED Leave Day'] = pat['ED Leave'] // (24*60)
pat['ED LoS'] = pat['ED Leave'] - pat['ED Arr']
pat['<4hr ED'] = np.where(~pat['ED Arr'].isna(), pat['ED LoS'] <= 240, np.nan)

ed4hr = (pat.groupby('ED Leave Day', as_index=False)['<4hr ED'].value_counts()
         .pivot(index='ED Leave Day', columns='<4hr ED', values='count'))
ed4hr['4hr Performance'] = ed4hr[1] / (ed4hr[0] + ed4hr[1]) * 100

fig = plt.figure(figsize=(15, 10))
ed4hr['4hr Performance'].plot(fig=fig, title='ED 4hr Performance', ylabel='4hr Performance', color='red')
plt.savefig(f'Plots/4hr Performance.png', bbox_inches='tight')
plt.close()

print('------4HR PERFORMANCE------')
print(f'Average 4hr Performance: {ed4hr['4hr Performance'].mean():.2f}%')

###################################################################################################
####OCCUPANCY
###################################################################################################
occ_cols = [col for col in occ.columns if 'Occ' in col]
occ_metrics = (occ.groupby('Day', observed=False)
               .agg({'ED Occ':['min', q25, 'mean', q75, 'max'],
                     'MRU Occ':['min', q25, 'mean', q75, 'max'],
                     'SAU Occ':['min', q25, 'mean', q75, 'max'],
                     'SDEC Occ':['min', q25, 'mean', q75, 'max'],
                     'FSDEC Occ':['min', q25, 'mean', q75, 'max'],
                     'Spec Occ':['min', q25, 'mean', q75, 'max']}))

fig, axs = plt.subplots(3, 2, figsize=(30, 20), sharex='col', layout='constrained')
for ax, col in zip(axs.reshape(-1), occ_cols): 
    data = occ_metrics[col].copy()
    ax.plot(occ_metrics.index, data['mean'].fillna(0), '-r', label='Mean')
    ax.fill_between(occ_metrics.index, data['min'].fillna(0), data['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    ax.fill_between(occ_metrics.index, data['q25'].fillna(0), data['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    ax.set_title(col.split(' ')[0].replace('Spec', 'All Specialties'), fontsize=18)
    ax.tick_params(axis='both',  which='major', labelsize=16)
    if col == 'ED Occ':
        ax.legend(fontsize=16)
fig.supxlabel('Day', fontsize=16)
fig.supylabel('Occupancy', fontsize=16)
fig.suptitle('Occupancy',fontsize=20)
plt.savefig('Plots/Overall Occupancy.png', bbox_inches='tight')
plt.close()

av_occ = occ[occ_cols].agg(['min', q25, 'mean', q75, 'max']).round()
print('------OCCUPANCY-------')
print('Overall:')
print(av_occ)

######################Specialty Occupancy
#Filter out rows where pat doesn't go to specialty
specialty = pat.loc[~pat['Specialty'].isna()].copy()
#Get every sample time that a patient was in the model
specialty['sample'] = specialty.apply(lambda row: list(range(math.ceil(row['Spec Arr']/60)*60, int(row['Spec Dis']), 60)), axis=1)
#Explode out and group up
specialty = specialty[['Specialty', 'sample']].explode('sample').value_counts().reset_index()
specialty['count'] = specialty['count'] / default_params.iterations
specialty = specialty.pivot(index='sample', columns='Specialty', values='count').sort_values(by='sample')
specialties = specialty.columns
specialty['Day'] = specialty.index // (24*60)

#Make a plot for each one
os.makedirs('Plots/Specialty Occupancy', exist_ok=True)
for spec in specialties:
    occ_metrics = specialty[['Day', spec]].groupby('Day').agg(['min', q25, 'mean', q75, 'max'])[spec]
    fig = plt.figure(figsize=(15, 10))
    plt.plot(occ_metrics.index, occ_metrics['mean'].fillna(0), '-r', label='Mean')
    plt.fill_between(occ_metrics.index, occ_metrics['min'].fillna(0), occ_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    plt.fill_between(occ_metrics.index, occ_metrics['q25'].fillna(0), occ_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    plt.title(spec, fontsize=18)
    plt.tick_params(axis='both',  which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Occupancy', fontsize=16)
    plt.savefig(f'Plots/Specialty Occupancy/{spec}.png', bbox_inches='tight')
    plt.close()

spec_occ = specialty.agg(['min', q25, 'mean', q75, 'max']).round().T
print('Specialty:')
print(spec_occ.sort_values(by='mean', ascending=False).head(5))

###################################################################################################
####QUEUES
###################################################################################################
que_cols = [col for col in occ.columns if 'Queue' in col]
que_metrics = (occ.groupby('Day', observed=False)
               .agg({'ED Queue':['min', q25, 'mean', q75, 'max'],
                     'MRU Queue':['min', q25, 'mean', q75, 'max'],
                     'SAU Queue':['min', q25, 'mean', q75, 'max'],
                     'SDEC Queue':['min', q25, 'mean', q75, 'max'],
                     'FSDEC Queue':['min', q25, 'mean', q75, 'max'],
                     'Spec Queue':['min', q25, 'mean', q75, 'max']}))

fig, axs = plt.subplots(3, 2, figsize=(30, 20), sharex='col', layout='constrained')
for ax, col in zip(axs.reshape(-1), que_cols): 
    data = que_metrics[col].copy()
    ax.plot(que_metrics.index, data['mean'].fillna(0), '-r', label='Mean')
    ax.fill_between(que_metrics.index, data['min'].fillna(0), data['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    ax.fill_between(que_metrics.index, data['q25'].fillna(0), data['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    ax.set_title(col.split(' ')[0].replace('Spec', 'All Specialties'), fontsize=18)
    ax.tick_params(axis='both',  which='major', labelsize=16)
    if col == 'ED Occ':
        ax.legend(fontsize=16)
fig.supxlabel('Day', fontsize=16)
fig.supylabel('Queue', fontsize=16)
fig.suptitle('Queues', fontsize=20)
plt.savefig('Plots/Area Queues.png', bbox_inches='tight')
plt.close()

av_que = occ[que_cols].agg(['min', q25, 'mean', q75, 'max']).round()
print('------QUEUES-------')
print('Overall:')
print(av_que)

###################################################################################################
####LoS
###################################################################################################
pat['Arrival Day'] = pat['Model Arr'] // (60*24)
#Calculate leave times (inc queue for next location). ED calculated earlier
pat['MRU or SAU Leave'] = np.where(~pat['MRU or SAU Arr'].isna(), pat['SDEC Arr'].fillna(pat['Spec Arr']).fillna(pat['Home']), np.nan)
pat['SDEC Leave'] = np.where(~pat['SDEC Arr'].isna(), pat['Spec Arr'].fillna(pat['Home']), np.nan)

#Calculate LoS
pat['MRU LoS'] = np.where(pat['MRU or SAU'] == 'MRU', pat['MRU or SAU Leave'] - pat['MRU or SAU Arr'], np.nan)
pat['SAU LoS'] = np.where(pat['MRU or SAU'] == 'SAU', pat['MRU or SAU Leave'] - pat['MRU or SAU Arr'], np.nan)
pat['SDEC LoS'] = np.where(pat['(F)SDEC'] == 'SDEC', pat['SDEC Leave'] - pat['SDEC Arr'], np.nan)
pat['FSDEC LoS'] = np.where(pat['(F)SDEC'] == 'FSDEC', pat['SDEC Leave'] - pat['SDEC Arr'], np.nan)
pat['Spec LoS'] = pat['Home'] - pat['Spec Arr']

#Summary metrics
los_cols = [col for col in pat.columns if 'LoS' in col]
los_metrics = (pat.groupby('Arrival Day', observed=False)
               .agg({'ED LoS':['min', q25, 'mean', q75, 'max'],
                     'MRU LoS':['min', q25, 'mean', q75, 'max'],
                     'SAU LoS':['min', q25, 'mean', q75, 'max'],
                     'SDEC LoS':['min', q25, 'mean', q75, 'max'],
                     'FSDEC LoS':['min', q25, 'mean', q75, 'max'],
                     'Spec LoS':['min', q25, 'mean', q75, 'max']}))

#plot
fig, axs = plt.subplots(3, 2, figsize=(30, 20), sharex='col', layout='constrained')
for ax, col in zip(axs.reshape(-1), los_cols): 
    data = los_metrics[col].copy()
    ax.plot(los_metrics.index, data['mean'].fillna(0), '-r', label='Mean')
    ax.fill_between(los_metrics.index, data['min'].fillna(0), data['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    ax.fill_between(los_metrics.index, data['q25'].fillna(0), data['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    ax.set_title(col.split(' ')[0].replace('Spec', 'All Specialties'), fontsize=18)
    ax.tick_params(axis='both',  which='major', labelsize=16)
    if col == 'ED LoS':
        ax.legend(fontsize=16)
fig.supxlabel('Day', fontsize=16)
fig.supylabel('LoS', fontsize=16)
fig.suptitle('Length of Stay', fontsize=20)
plt.savefig('Plots/Length of Stay.png', bbox_inches='tight')
plt.close()

av_los = pat[los_cols].agg(['min', q25, 'mean', q75, 'max']).round()
print('------LoS-------')
print('Overall:')
print(av_los)

###################################################################################################
####Arrivals
###################################################################################################
transitions = pd.Series([(p[i], p[i+1]) for p in pat['Path'] for i in range(len(p)-1)]
                            ).value_counts().reset_index()
transitions.columns = ["pair", "count"]
transitions[["from", "to"]] = pd.DataFrame(transitions["pair"].tolist())
transitions["prob"] = transitions["count"] / transitions.groupby("from")["count"].transform("sum")

pathways = (default_params.transitions[['from', 'to', 'prob']]
            .merge(transitions[['from', 'to', 'prob']], on=['from', 'to'], how='outer',
                   suffixes=[' Input', ' Output']))

pathways['diff'] = abs(pathways['prob Input'] - pathways['prob Output'])
print('------PATHWAYS------')
print(pathways.sort_values(by='diff', ascending=False))
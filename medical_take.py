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
from specialty_bed_optimisation import optimise_beds 

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
    iterations = 5#0
    sample_time = 1
    print_outs = False
    occ_sample_time = 60
    pat_res = []
    occ_res = []
    spec_res = []

    #########Settings
    #True if infinite capacities, false if restrictions
    inf_model = False
    #(only relevant if above is false), if true, input total number of specialty beds and distribute
    opt_spec = True
    no_beds = 850
    #True if user defined pathway probabilities
    set_pathways = False

    #########Read in the data
    conn_str = open('G:/PerfInfo/Performance Management/OR Team/Useful Files/uhpbi conn string.txt', 'r')
    uhpbi_engine = create_engine(conn_str.read())
    conn_str.close()
    #Get query and read in the data
    SQL = open('G:/PerfInfo/Performance Management/OR Team/Emily Projects/Discrete Event Simulation/Medical Take/SQL/FullHospitalModel.sql', 'r')
    df = pd.read_sql_query(SQL.read(), uhpbi_engine)
    SQL.close()
    uhpbi_engine.dispose()

    #########Specialties
    specialties = df['PerformanceSpecialty'].drop_duplicates().tolist()
    #Specialty probabilities
    spec = df['PerformanceSpecialty'].value_counts()
    spec = spec / spec.sum()

    #########timings
    ED_LoS = 4*60
    MRU_LoS = 12*60
    SAU_LoS = 6*60
    SDEC_FSDEC_LoS = 6*60
    Spec_LoS = df.groupby('PerformanceSpecialty')['Ward_LoS'].mean()

    #########Location Capacities
    if inf_model:
        ED_beds = MRU_beds = SAU_beds = SDEC_beds = FSDEC_beds = np.inf
        spec_beds = pd.DataFrame({'Specialty':specialties, 'No_beds':[np.inf]*len(specialties)})
    else:
        ED_beds = 105
        MRU_beds = 47
        SAU_beds = 19
        MSDEC_beds = 45
        FSDEC_beds = 10
        if opt_spec:
            spec_beds = optimise_beds(df[['PerformanceSpecialty', 'Ward_arr', 'Ward_LoS']].copy().dropna().copy(), no_beds)
        else:
            spec_beds = pd.read_excel('G:/PerfInfo/Performance Management/OR Team/Emily Projects/Discrete Event Simulation/Medical Take/Inputs/Specialty Capacities.xlsx')

    #########Pathways
    #Get the probabilities of each pathway in the data
    def build_path(row):
        path = ['start', row['Arrival']]
        for col in ["MRU or SAU", "MSDEC", "Spec Ward"]:
            if pd.notna(row[col]) and row[col] != "":
                path.append(row[col])
        path.append("Home")
        return path

    if not set_pathways:
        paths = df.apply(build_path, axis=1)
        transitions = pd.Series([(p[i], p[i+1]) for p in paths for i in range(len(p)-1)]
                                ).value_counts().reset_index()
        transitions.columns = ["pair", "count"]
        transitions[["from", "to"]] = pd.DataFrame(transitions["pair"].tolist())
        transitions["prob"] = transitions["count"] / transitions.groupby("from")["count"].transform("sum")
    else:
        transitions = pd.read_excel('G:/PerfInfo/Performance Management/OR Team/Emily Projects/Discrete Event Simulation/Medical Take/Inputs/Pathway Probabilities.xlsx')
        check = transitions.groupby('from')['prob'].sum().round()
        if len(check.loc[check != 1.0]) > 0:
            print('ERROR!!!! PATHWAY PROBABILITIES DO NOT SUM TO 1')

    # Route lookup disctionary
    routing = defaultdict(lambda: ([], []))
    for _, r in transitions.iterrows():
        routing[r["from"]][0].append(r["to"])
        routing[r["from"]][1].append(r["prob"])

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
        self.FSDEC_or_MSDEC = next((i for i in self.path if 'FSDEC' in i or 'MSDEC' in i), None)
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
        self.spec_occupancy_results = []
        #start environment, set patient counter to 0 and set run number
        self.env = simpy.Environment()
        self.input_params = input_params
        self.patient_counter = 0
        self.run_number = run_number

        #establish location resources
        self.ED = simpy.PriorityResource(self.env, capacity=input_params.ED_beds)
        self.MRU = simpy.PriorityResource(self.env, capacity=input_params.MRU_beds)
        self.SAU = simpy.PriorityResource(self.env, capacity=input_params.SAU_beds)
        self.MSDEC = simpy.PriorityResource(self.env, capacity=input_params.MSDEC_beds)
        self.FSDEC = simpy.PriorityResource(self.env, capacity=input_params.FSDEC_beds)
        self.spec_resources = {area : simpy.PriorityResource(self.env, capacity=cap)
                               for area, cap in zip(input_params.spec_beds['Specialty'], input_params.spec_beds['No_beds'])}

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

        #To ensure the patient holds on to the previous bed while waiting for the next one, we must
        # record the resource and request for the space currently occupied.
        current_resource = None
        current_request = None
        current_departure_field = None

        # For each location:
        # resource, queue timestamp, arrival timestamp,
        # departure timestamp, mean length of stay.
        locations = {'ED': (self.ED, 'ED_queue', 'ED_arr', 'ED_dis',
                            self.input_params.ED_LoS),
                    'MRU': (self.MRU, 'MoS_queue', 'MoS_arr', 'MoS_dis',
                            self.input_params.MRU_LoS),
                    'SAU': (self.SAU, 'MoS_queue', 'MoS_arr', 'MoS_dis',
                            self.input_params.SAU_LoS),
                    'MSDEC': (self.MSDEC, 'SDEC_queue', 'SDEC_arr', 'SDEC_dis',
                             self.input_params.SDEC_FSDEC_LoS),
                    'FSDEC': (self.FSDEC, 'SDEC_queue', 'SDEC_arr', 'SDEC_dis',
                              self.input_params.SDEC_FSDEC_LoS)}

        #Loop over each patient pathway
        for loc in pat.path:
            #If direct, proceed to next location
            if loc == 'Direct':
                continue
            #If home, end the loop and record the final timings
            if loc == 'Home':
                break
            #If specialty ward, determine which one and LoS in the same way as above dict
            if loc == 'Specialty Ward':
                next_resource = self.spec_resources[pat.Specialty]
                queue_field = 'Spec_queue'
                arrival_field = 'Spec_arr'
                departure_field = 'Spec_dis'
                mean_los = self.input_params.Spec_LoS.loc[pat.Specialty]
            #If other location, get the right values for that from above dict
            elif loc in locations:
                (next_resource, queue_field, arrival_field, departure_field,
                 mean_los) = locations[loc]
            else:
                raise ValueError(f"Unrecognised patient location: {loc!r}")

            # The patient is ready for this destination.
            #Record the patient's start queueing time and request the next resource
            setattr(pat, queue_field, self.env.now)
            next_request = next_resource.request()
            yield next_request

            # The destination space is now available.
            # Release the previous space at the time of transfer.
            if current_resource is not None:
                #Record the time leaving previous location
                setattr(pat, current_departure_field, self.env.now)
                current_resource.release(current_request)

            # Record entry and retain this space.
            setattr(pat, arrival_field, self.env.now)

            #Update current resource/request
            current_resource = next_resource
            current_request = next_request
            current_departure_field = departure_field

            # Time in this area before becoming ready to move on.
            sampled_los = round(random.expovariate(1.0 / mean_los))
            yield self.env.timeout(sampled_los)

        # Discharge home: release the final occupied space.
        if current_resource is not None:
            setattr(pat, current_departure_field, self.env.now)
            current_resource.release(current_request)

        #Store overall results
        pat.model_dis = self.env.now
        self.store_patient_results(pat)

    def store_patient_results(self, patient):
        self.patient_results.append({'Run':self.run_number,
                                     'Pat':patient.id,
                                     'Path':patient.path,
                                     'Model Arr':patient.model_arr,
                                     'ED Queue':patient.ED_queue,
                                     'ED Arr':patient.ED_arr,
                                     'ED Dis':patient.ED_dis,
                                     'MRU or SAU':patient.MRU_or_SAU,
                                     'MRU or SAU Queue':patient.MoS_queue,
                                     'MRU or SAU Arr':patient.MoS_arr,
                                     'MRU or SAU Dis':patient.MoS_dis,
                                     '(F)SDEC':patient.FSDEC_or_MSDEC,
                                     'SDEC Queue':patient.SDEC_queue,
                                     'SDEC Arr':patient.SDEC_arr,
                                     'SDEC Dis':patient.SDEC_dis,
                                     'Specialty':patient.Specialty,
                                     'Spec Queue':patient.Spec_queue,
                                     'Spec Arr':patient.Spec_arr,
                                     'Spec Dis':patient.Spec_dis,
                                     'Home':patient.model_dis})

    def store_occupancy(self):
        while True:
            #Get overall specialty queue and occ
            spec_queue = sum(len(res.queue) for res in self.spec_resources.values())
            spec_occ = sum(res.count for res in self.spec_resources.values())

            self.occupancy_results.append({'Run':self.run_number,
                                            'Time':self.ED._env.now,
                                            'ED Queue':len(self.ED.queue),
                                            'ED Occ':self.ED.count,
                                            'MRU Queue':len(self.MRU.queue),
                                            'MRU Occ':self.MRU.count,
                                            'SAU Queue':len(self.SAU.queue),
                                            'SAU Occ':self.SAU.count,
                                            'MSDEC Queue':len(self.MSDEC.queue),
                                            'MSDEC Occ':self.MSDEC.count,
                                            'FSDEC Queue':len(self.FSDEC.queue),
                                            'FSDEC Occ':self.FSDEC.count,
                                            'Spec Queue':spec_queue,
                                            'Spec Occ':spec_occ})
            
            for spec, res in self.spec_resources.items():
                self.spec_occupancy_results.append({'run':self.run_number,
                                                    'Time':self.env.now,
                                                    'specialty':spec,
                                                    'queue':len(res.queue),
                                                    'occ':res.count,})

            yield self.env.timeout(self.input_params.occ_sample_time)

    def run(self):
        self.env.process(self.arrivals())
        self.env.process(self.store_occupancy())
        self.env.run(until = self.input_params.run_time)
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.occupancy_results
        default_params.spec_res += self.spec_occupancy_results
        return self.patient_results, self.occupancy_results, self.spec_occupancy_results

def export_results(pat_results, occ_results, spec_results):
    #Patient
    pat_df = pd.DataFrame(pat_results)
    #Occuancy
    occ_df = pd.DataFrame(occ_results)
    occ_df['Day'] = (occ_df['Time'] // (24*60)) - 20
    #Specialty occupancy
    spec_occ_df = pd.DataFrame(spec_results)
    spec_occ_df['Day'] = (spec_occ_df['Time'] // (24*60)) - 20
    return pat_df, occ_df, spec_occ_df

def run_the_model(input_params):
    #run the model for the number of iterations specified
    for run in range(input_params.iterations):
        print(f"Run {run+1} of {input_params.iterations}")
        model = Medical_Take_Model(run, input_params)
        model.run()
    patient_df, occ_df, spec_occ_df = export_results(input_params.pat_res,
                                                     input_params.occ_res,
                                                     input_params.spec_res)
    return patient_df, occ_df, spec_occ_df

pat, occ, spec = run_the_model(default_params)
#Create results folder if not exist
path = f'G:/PerfInfo/Performance Management/OR Team/Emily Projects/Discrete Event Simulation/Medical Take/Results/{default_params.run_name}'
os.makedirs(path, exist_ok=True)  # create if it doesn't exist (including parent dirs)
os.chdir(path)

for file in ['Full Outputs', 'Plots']:
    os.makedirs(file, exist_ok=True)

#Save full data
pat.to_csv(f'Full Outputs/patients - {default_params.run_name}.csv')
occ.to_csv(f'Full Outputs/occupancy - {default_params.run_name}.csv')
spec.to_csv(f'Full Outputs/specialty - {default_params.run_name}.csv')


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
spec = spec.loc[spec['Day'] > 0].copy()

###################################################################################################
####4hr Performance
###################################################################################################
pat['ED Leave Day'] = pat['ED Dis'] // (24*60)
pat['ED LoS'] = pat['ED Dis'] - pat['ED Arr']
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
                     'MSDEC Occ':['min', q25, 'mean', q75, 'max'],
                     'FSDEC Occ':['min', q25, 'mean', q75, 'max'],
                     'Spec Occ':['min', q25, 'mean', q75, 'max']}))

fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex='col', layout='constrained')
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
#Average and aggregate over runs and daily variation
specialty = spec.groupby(['Day', 'specialty'], as_index=False)[['queue', 'occ']].agg(['min', q25, 'mean', q75, 'max'])

#Make a plot for each one
os.makedirs('Plots/Specialty', exist_ok=True)
for spec_name in specialty['specialty'].drop_duplicates().tolist():
    data = specialty.loc[specialty['specialty'] == spec_name].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), layout='constrained')

    #queue
    df_queue = data['queue']
    ax1.plot(data['Day'], df_queue['mean'].fillna(0), '-r', label='Mean')
    ax1.fill_between(data['Day'], df_queue['min'].fillna(0), df_queue['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    ax1.fill_between(data['Day'], df_queue['q25'].fillna(0), df_queue['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    ax1.set_title('Queue', fontsize=18)
    ax1.tick_params(axis='both',  which='major', labelsize=16)
    ax1.legend(fontsize=16)
    ax1.set_xlabel('Day', fontsize=16)
    ax1.set_ylabel('Queue Length', fontsize=16)

    #Occupancy
    df_occ = data['occ']
    ax2.plot(data['Day'], df_occ['mean'].fillna(0), '-r', label='Mean')
    ax2.fill_between(data['Day'], df_occ['min'].fillna(0), df_occ['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    ax2.fill_between(data['Day'], df_occ['q25'].fillna(0), df_occ['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    ax2.set_title('Occupancy', fontsize=18)
    ax2.tick_params(axis='both',  which='major', labelsize=16)
    ax2.set_xlabel('Day', fontsize=16)
    ax2.set_ylabel('Occupancy', fontsize=16)

    fig.suptitle(spec_name, fontsize=26)
    bed_cap = default_params.spec_beds.loc[default_params.spec_beds['Specialty'] == spec_name, 'No_beds'].iloc[0]
    fig.text(0, 1, f'Bed Capacity: {bed_cap}', fontsize=20)

    plt.savefig(f'Plots/Specialty/{spec_name.replace('&', '')}.png', bbox_inches='tight')
    plt.close()


print('Specialty:')
print(specialty.groupby('specialty').mean()['occ'].sort_values(by='mean', ascending=False).head(5))

###################################################################################################
####QUEUES
###################################################################################################
que_cols = [col for col in occ.columns if 'Queue' in col]
que_metrics = (occ.groupby('Day', observed=False)
               .agg({'ED Queue':['min', q25, 'mean', q75, 'max'],
                     'MRU Queue':['min', q25, 'mean', q75, 'max'],
                     'SAU Queue':['min', q25, 'mean', q75, 'max'],
                     'MSDEC Queue':['min', q25, 'mean', q75, 'max'],
                     'FSDEC Queue':['min', q25, 'mean', q75, 'max'],
                     'Spec Queue':['min', q25, 'mean', q75, 'max']}))

fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex='col', layout='constrained')
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
####Queue LoS
###################################################################################################
pat['Arrival Day'] = pat['Model Arr'] // (60*24)
#Calculate LoS
pat['ED Queue LoS'] = pat['ED Dis'] - pat['ED Queue']
pat['MRU Queue LoS'] = np.where(pat['MRU or SAU'] == 'MRU', pat['MRU or SAU Dis'] - pat['MRU or SAU Queue'], np.nan)
pat['SAU Queue LoS'] = np.where(pat['MRU or SAU'] == 'SAU', pat['MRU or SAU Dis'] - pat['MRU or SAU Queue'], np.nan)
pat['MSDEC Queue LoS'] = np.where(pat['(F)SDEC'] == 'MSDEC', pat['SDEC Dis'] - pat['SDEC Queue'], np.nan)
pat['FSDEC Queue LoS'] = np.where(pat['(F)SDEC'] == 'FSDEC', pat['SDEC Dis'] - pat['SDEC Queue'], np.nan)
pat['Spec Queue LoS'] = pat['Spec Dis'] - pat['Spec Queue']
queue_cols = ['ED Queue LoS', 'MRU Queue LoS', 'SAU Queue LoS', 'MSDEC Queue LoS', 'FSDEC Queue LoS', 'Spec Queue LoS']

#Summary metrics
los_metrics = (pat.groupby('Arrival Day', observed=False)
               .agg({'ED Queue LoS':['min', q25, 'mean', q75, 'max'],
                     'MRU Queue LoS':['min', q25, 'mean', q75, 'max'],
                     'SAU Queue LoS':['min', q25, 'mean', q75, 'max'],
                     'MSDEC Queue LoS':['min', q25, 'mean', q75, 'max'],
                     'FSDEC Queue LoS':['min', q25, 'mean', q75, 'max'],
                     'Spec Queue LoS':['min', q25, 'mean', q75, 'max']}))

#plot
fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex='col', layout='constrained')
for ax, col in zip(axs.reshape(-1), queue_cols): 
    data = los_metrics[col].copy()
    ax.plot(los_metrics.index, data['mean'].fillna(0), '-r', label='Mean')
    ax.fill_between(los_metrics.index, data['min'].fillna(0), data['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
    ax.fill_between(los_metrics.index, data['q25'].fillna(0), data['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
    ax.set_title(col.split(' ')[0].replace('Spec', 'All Specialties'), fontsize=18)
    ax.tick_params(axis='both',  which='major', labelsize=16)
    if col == 'ED Queue LoS':
        ax.legend(fontsize=16)
fig.supxlabel('Day', fontsize=16)
fig.supylabel('LoS', fontsize=16)
fig.suptitle('Time spent waiting for next location', fontsize=20)
plt.savefig('Plots/Time for next loc.png', bbox_inches='tight')
plt.close()

av_los = pat[queue_cols].agg(['min', q25, 'mean', q75, 'max']).round()
print('------Time Queueing for next loc-------')
print('Overall:')
print(av_los)

###################################################################################################
####LoS
###################################################################################################
pat['Arrival Day'] = pat['Model Arr'] // (60*24)
#Calculate LoS
pat['MRU LoS'] = np.where(pat['MRU or SAU'] == 'MRU', pat['MRU or SAU Dis'] - pat['MRU or SAU Arr'], np.nan)
pat['SAU LoS'] = np.where(pat['MRU or SAU'] == 'SAU', pat['MRU or SAU Dis'] - pat['MRU or SAU Arr'], np.nan)
pat['MSDEC LoS'] = np.where(pat['(F)SDEC'] == 'MSDEC', pat['SDEC Dis'] - pat['SDEC Arr'], np.nan)
pat['FSDEC LoS'] = np.where(pat['(F)SDEC'] == 'FSDEC', pat['SDEC Dis'] - pat['SDEC Arr'], np.nan)
pat['Spec LoS'] = pat['Spec Dis'] - pat['Spec Arr']


#Summary metrics
los_cols = ['ED LoS', 'MRU LoS', 'SAU LoS', 'MSDEC LoS', 'FSDEC LoS', 'Spec LoS']
los_metrics = (pat.groupby('Arrival Day', observed=False)
               .agg({'ED LoS':['min', q25, 'mean', q75, 'max'],
                     'MRU LoS':['min', q25, 'mean', q75, 'max'],
                     'SAU LoS':['min', q25, 'mean', q75, 'max'],
                     'MSDEC LoS':['min', q25, 'mean', q75, 'max'],
                     'FSDEC LoS':['min', q25, 'mean', q75, 'max'],
                     'Spec LoS':['min', q25, 'mean', q75, 'max']}))

#plot
fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex='col', layout='constrained')
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
####Pathways
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

###################################################################################################
####Arrivals
###################################################################################################
daily_arrs = []
for inp, out in [('ED_arr', 'ED Arr'), ('SDEC_arr', 'SDEC Arr'),
                 ('MRU_SAU_arr', 'MRU or SAU Arr'), ('Ward_arr', 'Spec Arr')]:
    data_arr = default_params.df[inp].dropna().dt.date.value_counts().mean()
    outp_arr = (pat[out].dropna() // (60*24)).value_counts().mean() / default_params.iterations
    daily_arrs.append([out, data_arr, outp_arr, data_arr-outp_arr])

print('-----ARRIVALS------')
print(pd.DataFrame(daily_arrs, columns=['Location', 'Input', 'Output', 'Diff']))



#specialty arrivals
inp = default_params.df[['PerformanceSpecialty', 'Ward_arr']].dropna()
inp['Ward_arr'] = inp['Ward_arr'].dt.date
all_dates = pd.DataFrame(itertools.product(inp['Ward_arr'].drop_duplicates(),
                                            inp['PerformanceSpecialty'].drop_duplicates()),
                            columns=['Ward_arr', 'PerformanceSpecialty'])
#merge onto all dates and fill blanks with 0. Group up to get mean daily arr by specialty.
inp_arr = (all_dates.merge(inp[['PerformanceSpecialty', 'Ward_arr']].value_counts().reset_index(),
                        on=['Ward_arr', 'PerformanceSpecialty'], how='left').fillna(0))
inp_arr = inp_arr.groupby('PerformanceSpecialty')['count'].mean().rename('Ward_arr')
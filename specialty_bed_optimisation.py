import numpy as np
import pandas as pd
import itertools


def optimise_beds(df, no_beds):
    #Average LoS per specialty (in days)
    los = df.groupby('PerformanceSpecialty')['Ward_LoS'].mean() / (60*24)

    #Average daily arrivals. Need to merge on all dates to conider days where no arrivals for smaller
    #specialties
    df['Ward_arr'] = df['Ward_arr'].dt.date
    all_dates = pd.DataFrame(itertools.product(df['Ward_arr'].drop_duplicates(),
                                               df['PerformanceSpecialty'].drop_duplicates()),
                            columns=['Ward_arr', 'PerformanceSpecialty'])
    #merge onto all dates and fill blanks with 0. Group up to get mean daily arr by specialty.
    arr = (all_dates.merge(df[['PerformanceSpecialty', 'Ward_arr']].value_counts().reset_index(),
                        on=['Ward_arr', 'PerformanceSpecialty'], how='left').fillna(0))
    arr = arr.groupby('PerformanceSpecialty')['count'].mean().rename('Ward_arr')

    #Combine los and arrivals into one and calculate the expected occupancy (littles law)
    data = pd.DataFrame([arr, los]).T.dropna()
    data['Exp Occ'] = data['Ward_arr'] * data['Ward_LoS']

    #####Use these expected occupancies to scale to meet overall bed limits.
    #Anywhere that littles law suggests less than a single bed should have 1 bed.
    single_bed_mask = data['Exp Occ'] < 1
    #Create no_beds column, populate with the single beds first
    data['init_beds'] = np.where(single_bed_mask, 1.0, 0.0)

    #distribute the rest using the specialty proportions
    remaining_beds = no_beds - data['init_beds'].sum()
    remaining_props = data.loc[~single_bed_mask, 'Exp Occ'].sum()

    #Work out how many beds each specialty needs
    data.loc[~single_bed_mask, 'init_beds'] =  (data.loc[~single_bed_mask, 'Exp Occ'] / remaining_props) * remaining_beds

    #Rounding the above could lead to under or over utilisting the total number of beds, do this to mitigate
    #Allocate whole beds initially
    data.loc[~single_bed_mask, "Allocated"] = (np.floor(data.loc[~single_bed_mask, "init_beds"]).astype(int))
    data['Allocated'] = data['Allocated'].fillna(1)#baseline single beds

    # Work out how many beds are left due to rounding
    beds_left = no_beds - data["Allocated"].sum()

    # Give those beds to wards with the largest decimal remainder
    indices = (data.loc[~single_bed_mask, "init_beds"] % 1).nlargest(int(beds_left)).index
    data.loc[indices, "Allocated"] += 1

    #Return and rename required columns
    data = data.reset_index()[['PerformanceSpecialty', 'Allocated']]
    data.columns = ['Specialty', 'No_beds']
    return data

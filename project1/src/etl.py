import utils
import numpy as np
import pandas as pd
import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''

    dead_patient_indx_date = mortality.filter(['patient_id','timestamp'], axis=1)
    dead_patient_indx_date['indx_date'] = dead_patient_indx_date['timestamp'].apply(lambda x: (pd.to_datetime(x) - pd.Timedelta(30, unit='D'))).astype(str)
    dead_patient_indx_date = dead_patient_indx_date.drop(columns=['timestamp'], axis=1)
    print(dead_patient_indx_date)

    alive_patients = -events['patient_id'].isin(mortality['patient_id'])
    alive_patient_events = events[alive_patients]
    alive_patient_indx_date = alive_patient_events.groupby('patient_id')['timestamp'].agg({'timestamp':'max'}).rename(columns = {'timestamp' : 'indx_date'}).reset_index()
    print(alive_patient_indx_date)

    indx_date = pd.concat([dead_patient_indx_date, alive_patient_indx_date], sort=False)
    print(indx_date)
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    patient_event_indx_date = pd.merge(events, indx_date, how = 'inner', on ='patient_id')
    patient_event_indx_date['obs_window_lower_date'] = patient_event_indx_date['indx_date'].apply(lambda x: (pd.to_datetime(x) - pd.Timedelta(2000, unit='D'))).astype(str)
    filtered_events_all_columns = patient_event_indx_date[(patient_event_indx_date['timestamp'] >= patient_event_indx_date['obs_window_lower_date']) & (patient_event_indx_date['timestamp'] <= patient_event_indx_date['indx_date'])]
    filtered_events = filtered_events_all_columns.filter(['patient_id','event_id', 'value'], axis=1)
    print("filtered_events")
    print(filtered_events)
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''

    events_df_with_value = filtered_events_df.dropna(subset=['value'])
    diag_and_med_events = events_df_with_value[events_df_with_value['event_id'].str.startswith('DIAG') | events_df_with_value['event_id'].str.startswith('DRUG')]
    diag_and_med_events_agg = diag_and_med_events.groupby(['patient_id', 'event_id'])['value'].agg({'value':'sum'}).rename(columns = {'value' : 'feature_value'}).reset_index()

    lab_events = events_df_with_value[events_df_with_value['event_id'].str.startswith('LAB')]
    lab_events_agg = lab_events.groupby(['patient_id', 'event_id'])['value'].agg({'value':'count'}).rename(columns = {'value' : 'feature_value'}).reset_index()

    agg_events_df = pd.concat([diag_and_med_events_agg, lab_events_agg], sort=False)
    agg_feature_df = pd.merge(agg_events_df, feature_map_df, how ='inner', on='event_id').rename(columns = {'idx' : 'feature_id'})

    aggregated_features = agg_feature_df.filter(['patient_id','feature_id', 'feature_value'], axis=1)

    # TODO Min-Max Normalize aggregated values
    feature_max = aggregated_features.groupby('feature_id')['feature_value'].agg({'feature_value':'max'}).rename(columns = {'feature_value' : 'feature_value_max'}).reset_index()

    aggregated_events_norm = pd.merge(aggregated_features, feature_max, how ='inner', on='feature_id')
    aggregated_events_norm['feature_value_norm'] = aggregated_events_norm['feature_value']/aggregated_events_norm['feature_value_max']
    aggregated_events = aggregated_events_norm.filter(['patient_id','feature_id', 'feature_value_norm'], axis=1).rename(columns = {'feature_value_norm' : 'feature_value'})

    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''

    print("creating patient_features_dict")
    patient_features = {}
    for key, value in aggregated_events.groupby('patient_id'):
        patient_features[key] = list(zip(value['feature_id'], value['feature_value']))

    print("creating mortality_dict")
    unique_patients = pd.DataFrame(np.hstack(events['patient_id'].unique()), columns = ['patient_id'])
    all_patient_mortality = pd.merge(unique_patients, mortality, how='left', on='patient_id').drop(columns=['timestamp'], axis=1)
    all_patient_mortality['label'] = all_patient_mortality['label'].fillna(0)

    mortality = all_patient_mortality.set_index('patient_id')['label'].to_dict()
    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''

    file_out = ''
    deliverable_out = ''
    for key in sorted(patient_features.keys()):
        feature_tuples = patient_features.get(key)
        label = mortality.get(key)
        line = ''
        sorted_by_feature_id = sorted(feature_tuples, key=lambda tuple: tuple[0])
        for feature_id, feature_value in sorted_by_feature_id:
            line += ' ' + str(int(feature_id)) + ':' + '%.6f' % feature_value
        file_out += str(label) + line + ' \n'
        deliverable_out += str(int(key)) + ' ' + str(label) + line + ' \n'

    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    deliverable1.write(bytes(file_out,'UTF-8')); #Use 'UTF-8'
    deliverable2.write(bytes(deliverable_out,'UTF-8'))

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()

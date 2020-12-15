import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''

    dead_patient_events = events[events['patient_id'].isin(mortality['patient_id'])]
    grouped_dead_events = dead_patient_events.groupby('patient_id').size().to_frame(name='dead_event_count').reset_index()

    alive_patient_events = events[-events['patient_id'].isin(mortality['patient_id'])]
    grouped_alive_events = alive_patient_events.groupby('patient_id').size().to_frame(name='alive_event_count').reset_index()

    avg_dead_event_count = grouped_dead_events['dead_event_count'].mean()
    max_dead_event_count = grouped_dead_events['dead_event_count'].max()
    min_dead_event_count = grouped_dead_events['dead_event_count'].min()
    avg_alive_event_count = grouped_alive_events['alive_event_count'].mean()
    max_alive_event_count = grouped_alive_events['alive_event_count'].max()
    min_alive_event_count = grouped_alive_events['alive_event_count'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    dead_patient_events = events[events['patient_id'].isin(mortality['patient_id'])]
    dead_events_grouped_by_patient_and_date = dead_patient_events.groupby(['patient_id', 'timestamp']).size().to_frame(name='dead_event_count').reset_index()
    dead_encounter_grouped_by_patient = dead_events_grouped_by_patient_and_date.groupby('patient_id').size().to_frame(name='dead_encounter_count').reset_index()

    alive_patient_events = events[-events['patient_id'].isin(mortality['patient_id'])]
    alive_events_grouped_by_patient_and_date = alive_patient_events.groupby(['patient_id', 'timestamp']).size().to_frame(name='alive_event_count').reset_index()
    alive_encounter_grouped_by_patient = alive_events_grouped_by_patient_and_date.groupby('patient_id').size().to_frame(name='alive_encounter_count').reset_index()

    avg_dead_encounter_count = dead_encounter_grouped_by_patient['dead_encounter_count'].mean()
    max_dead_encounter_count = dead_encounter_grouped_by_patient['dead_encounter_count'].max()
    min_dead_encounter_count = dead_encounter_grouped_by_patient['dead_encounter_count'].min()
    avg_alive_encounter_count = alive_encounter_grouped_by_patient['alive_encounter_count'].mean()
    max_alive_encounter_count = alive_encounter_grouped_by_patient['alive_encounter_count'].max()
    min_alive_encounter_count = alive_encounter_grouped_by_patient['alive_encounter_count'].min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    dead_patient_events = events[events['patient_id'].isin(mortality['patient_id'])]
    dead_events_grouped_by_patient = dead_patient_events.groupby('patient_id')['timestamp'].agg(['max', 'min'])
    dead_events_grouped_by_patient = dead_events_grouped_by_patient.apply(pd.to_datetime)
    dead_events_grouped_by_patient['rec_len'] = (dead_events_grouped_by_patient['max'] - dead_events_grouped_by_patient['min']).dt.days

    alive_patient_events = events[-events['patient_id'].isin(mortality['patient_id'])]
    alive_events_grouped_by_patient = alive_patient_events.groupby('patient_id')['timestamp'].agg(['max', 'min'])
    alive_events_grouped_by_patient = alive_events_grouped_by_patient.apply(pd.to_datetime)
    alive_events_grouped_by_patient['rec_len'] = (alive_events_grouped_by_patient['max'] - alive_events_grouped_by_patient['min']).dt.days

    avg_dead_rec_len = dead_events_grouped_by_patient['rec_len'].mean()
    max_dead_rec_len = dead_events_grouped_by_patient['rec_len'].max()
    min_dead_rec_len = dead_events_grouped_by_patient['rec_len'].min()
    avg_alive_rec_len = alive_events_grouped_by_patient['rec_len'].mean()
    max_alive_rec_len = alive_events_grouped_by_patient['rec_len'].max()
    min_alive_rec_len = alive_events_grouped_by_patient['rec_len'].min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()

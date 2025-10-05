import os
import pickle
from parse_data import filter_qa_general, update_segment_interval, filter_qa_specific, gather_mcq
from collections import defaultdict
from utils import save_json, load_json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', help='', choices = ['general', 'interval', 'specific', 'final'], required=True)
    args = parser.parse_args()
    stage = args.stage
    save_dir = './results'
    task2qa = defaultdict(list)
    if stage == 'general':
        for stage in ['single', 'multi', 'full']:
            filtered_qas = filter_qa_general('./results', stage, True)
            for qa in filtered_qas:
                task2qa[qa['task']].append(qa)                
        # save_json(task2qa, './task2qa_general.json')
    elif stage == 'interval': 
        task2qa = load_json('./task2qa_general.json')
        new_task2qa = update_segment_interval(save_dir)
        for task in new_task2qa:
            task2qa[task] = new_task2qa[task]
        save_json(task2qa, './task2qa_interval.json')
    elif stage == 'specific':
        task2qa_interval = load_json('./task2qa_interval.json')
        task2qa = filter_qa_specific(task2qa_interval)
        save_json(task2qa, './task2qa_specific.json')
    elif stage == 'final':
        total_avail_list = list()
        total_failed_list = list()
        for stage in ['single', 'multi', 'full']:
            avail_list, failed_list = gather_mcq('./results', stage, time_format = 'second')
            total_avail_list += avail_list
            total_failed_list += failed_list
        from utils import remove_tone, transform_options
        for item in total_avail_list:
            if item['task'] =='task2':
                item['question'] = remove_tone(item['question'])
            item = transform_options(item)
            task2qa[item['task']].append(item)
        save_json(total_avail_list, './benchmark.json')
        save_json(task2qa, './task2qa_final.json')
        save_json(failed_list, './final_failed.json')
    total = 0
    for task in task2qa:
        total += len(task2qa[task])
        print(task, len(task2qa[task]))
    print('Total:', total)
import os
import json
from utils import load_jsonl, save_json, load_json
from collections import defaultdict
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

scene2task ={
    'single':['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8'],
    'multi':[ 'task9', 'task10', 'task11', 'task12', 'task13'],
    'full':['task15', 'task16', 'task17']
    }
ability2task = {
    'tem':['task1', 'task8', 'task10', 'task11'],
    'spa':['task2', 'task4', 'task5'],
    'long':['task9', 'task15'],
    'emo':['task6', 'task7', 'task16'],
    'plo':['task12', 'task13', 'task17'],
    }
task2modality = {
    "task1":['speech'],
    "task2":['vocal_traits'],
    # "task3":['speech'],
    "task4":['sound_event'],
    "task5":['sound_event'],
    "task6":['vocal_traits'],
    "task7":['music'],
    "task8":['speech'],
    "task9":['speech'],
    "task10":['speech', 'vocal_traits'],
    "task11":['speech'],
    "task12":['sound_event','music'],
    "task13":['speech'],
    "task15":['speech','vocal_traits','sound_event','music'],
    "task16":['music'],
    "task17":['speech'],
}

if __name__ == "__main__":
    model_list = ["qwen2.5omni", "videollama2", "videollama3", "vita1.5","salmonn", "qwen2.5vl", "llavavideo", 
                  "internvl", "kimiaudio", "onellm", "gpt4o", "qwen2audio", "gemini_api", "omnir1", "salmonno1",
                  "avicuna", "aurelia"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=["a", "v", "av", "vt"], help='', required=False, default = 'av')
    parser.add_argument('--model-name', choices=model_list, help='', required=True)
    parser.add_argument('--result-dir', help='', required=False, default = './results')
    parser.add_argument('--duration-limit', help='', required=False,type = float, default = -1)
    parser.add_argument('--eval-task',  help='', action = 'store_true')
    parser.add_argument('--eval-audio',  help='', action = 'store_true')
    parser.add_argument('--eval-scene',  help='', action = 'store_true')
    parser.add_argument('--eval-ability',  help='', action = 'store_true')
    parser.add_argument('--scene-result',  help='', action = 'store_true')
    parser.add_argument('--duration-result',  help='', action = 'store_true')
    parser.add_argument('--difficulty-result',  help='', action = 'store_true')
    args = parser.parse_args()
    modality = args.modality
    model = args.model_name
    eval_result_path = os.path.join(args.result_dir, f'./eval_results_{model}_{modality}.jsonl')
    data = load_jsonl(eval_result_path)
    # eval_result_path = os.path.join(args.result_dir, f'./eval_results_{model}_{modality}.json')
    # data = load_json(eval_result_path)
    avail_qids = load_json('./new_qid.json')
    data = [i for i in data if i['qid'] in avail_qids]
    if args.duration_limit > 0:
        data = [item for item in data if item['duration'] <= args.duration_limit]
    if args.eval_task:
        task2qa = defaultdict(list)
        overall_result = []
        for item in data:
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                task2qa[item['task']].append(res)
                overall_result.append(res)
        print(len(overall_result))
        print("average acc:",sum(overall_result)/len(overall_result))
        for task in task2modality:
            try:
                print(f"{task} acc: {sum(task2qa[task])/len(task2qa[task])}")
            except ZeroDivisionError:
                continue
    if args.eval_audio:
        audio2qa = defaultdict(list)
        # single2qa = defaultdict(list)
        # multi2qa = defaultdict(list)
        # full2qa = defaultdict(list)
        for item in data:
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                for audiotype in task2modality[item['task']]:
                    audio2qa[audiotype].append(res)
                # if item['task'] in scene2task['single']:
                #     for audiotype in task2modality[item['task']]:
                #         single2qa[audiotype].append(res)
                # if item['task'] in scene2task['multi']:
                #     for audiotype in task2modality[item['task']]:
                #         multi2qa[audiotype].append(res)
                # if item['task'] in scene2task['full']:
                #     for audiotype in task2modality[item['task']]:
                #         full2qa[audiotype].append(res)
        for audiotype in audio2qa:
            print(f"{audiotype} acc: {sum(audio2qa[audiotype])/len(audio2qa[audiotype])}")
    if args.eval_ability:
        ability2qa = {key:[] for key in ability2task.keys()}
        for item in data:
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                for ability_type in ability2task:
                    if item['task'] in ability2task[ability_type]:
                        ability2qa[ability_type].append(res)
        for ability_type in ability2qa:
            print(f"{ability_type} acc: {sum(ability2qa[ability_type])/len(ability2qa[ability_type])}")
    if args.eval_scene:
        scene2qa = {key:[] for key in scene2task.keys()}
        for item in data:
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                for scene_type in scene2task:
                    if item['task'] in scene2task[scene_type]:
                        scene2qa[scene_type].append(res)
        for scene_type in scene2qa:
            print(f"{scene_type} acc: {sum(scene2qa[scene_type])/len(scene2qa[scene_type])}")
    # for audiotype in single2qa:
    #     print(f"{audiotype} single acc: {sum(single2qa[audiotype])/len(single2qa[audiotype])}")
    # for audiotype in multi2qa:
    #     print(f"{audiotype} multi acc: {sum(multi2qa[audiotype])/len(multi2qa[audiotype])}")
    # for audiotype in full2qa:
    #     print(f"{audiotype} full acc: {sum(full2qa[audiotype])/len(full2qa[audiotype])}")
    if args.scene_result:
        qid2segments = load_json('qid2segments.json')
        segments2results = defaultdict(list)
        for item in data:
            if item['task'] not in ['task9', 'task10', 'task11', 'task12', 'task13']:
                continue
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                segments = qid2segments[item['qid']]
                segments2results[segments].append(res)
        save_json(segments2results,f'./scene_results/{model}_scene.json')
    if args.duration_result:
        from evaluation import get_segment_time
        duration2results = defaultdict(list)
        for item in data:
            if item['task'] not in ['task9', 'task10', 'task11', 'task12', 'task13']:
                continue
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                duration = item['duration']
                duration2results[duration].append(res)
        save_json(duration2results,f'./duration_results/{model}_duration.json')
    if args.difficulty_result:
        difficulty2qid = load_json('./questions_by_difficulty.json')
        difficulty2results = defaultdict(list)
        for item in data:
            if item['model_answer'] is not None:
                res = item['model_answer']==item['correct_prefix']
                # t = 0
                for diff in difficulty2qid:
                    if item['qid'] in difficulty2qid[diff]:
                        # t = 1
                        difficulty2results[diff].append(res)
                # if t == 0:
                #     difficulty2results['5'].append(res)
        for diff in difficulty2results:
            print(f"{diff} acc: {sum(difficulty2results[diff])/len(difficulty2results[diff])}")
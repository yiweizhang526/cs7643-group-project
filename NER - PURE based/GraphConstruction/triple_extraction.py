import numpy as np
import json
import copy
import os

#gold_docs = [json.loads(line) for line in open(json_file)]
json_file="F:\\ECpy\\learn\\Master\\22spring\\CS7643\\project\\data\\ace05\\json\\train.json"
triple=[]
for line in open(json_file):
    sens_all=[]
    each_example=json.loads(line)
    for each_sentence in each_example['sentences']:
        if each_sentence is not None:
            sens_all+=each_sentence
    #print(sens_all)
    for each_sen_relations in each_example['relations']:
        for each_relation in each_sen_relations:
            if each_relation is not None:
                head_entity=""
                tail_entity=""
                for hspan in range(each_relation[0],each_relation[1]+1):
                    head_entity+=sens_all[hspan]
                    head_entity+=' '
                for tspan in range(each_relation[2],each_relation[3]+1):
                    tail_entity+=sens_all[tspan]
                    tail_entity+=' '
                relation=each_relation[4]
                triple.append([head_entity,relation,tail_entity])

with open("F:\\ECpy\\learn\\Master\\22spring\\CS7643\\triple.txt",'a',encoding='utf-8') as f:
        for triples in triple:
            f.writelines(str(triples)+'\n')
    
f.close()
    
    
        
#!/usr/bin/python3

import re
import json
from os import listdir
from os.path import basename, splitext, isfile, join
from absl import app, flags
from neo4j import GraphDatabase
from typing import List
from string import Template
from .query import prc_query, ptp_query, dev_query
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

FLAGS = flags.FLAGS

"""
def add_options():
  flags.DEFINE_string('folder', default=None, help='path of the fold that contains input')
  flags.DEFINE_string('ELT', default=None, help='path of the fold that contains input')
  flags.DEFINE_string('host', default='bolt://103.6.49.76:7687', help='host')
  flags.DEFINE_string('user', default='neo4j', help='username')
  flags.DEFINE_string('password', default='neo4j', help='password')
  flags.DEFINE_string('db', default='test2', help='database')
"""

host = config.get('General', 'host')
password = config.get('General', 'password')
user = config.get('General', 'user')
db = config.get('General', 'db')

template = Template("""
In order to synthesis $elt_name, the following precursors are used $precursor. $procedure.
""")

def graph_query(name:str, driver):
    records, summary, keys = driver.execute_query(
        """
MATCH (a:ELT {name: $elt_name})
    CALL apoc.path.subgraphAll(a, {
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
      minLevel: 0,
      maxLevel: 100
  })
    YIELD nodes, relationships
    MATCH (b:SOP) WHERE b IN nodes WITH b, nodes, relationships, collect(DISTINCT split(b.ceid, " ")[0]) AS ids
    MATCH (c:NUM) WHERE c IN nodes AND NOT split(c.ceid, " ")[0] in ids WITH collect(c) AS blacklist
    MATCH (a:ELT {name: $elt_name})
    CALL apoc.path.subgraphAll(a, {
    relationshipFilter: "Instance_of>|Synthesis_with>|Next_Operation>|Precursor>|Attribute_of<|Device>|Amount_of<|Unit_of|>Condition|Number|Number_of|OperationTarget>|Participant>|REFERENCED_IN>",
    minLevel: 0,
    maxLevel: 100,
    blacklistNodes: blacklist
  })
    YIELD nodes, relationships
    RETURN nodes, relationships;
""", elt_name=name, database_=db)

    if records == []:
        return "No Synthesis Path found"
    start = start_temp(records, driver, name)
    nodes = records[0][0]
    relations = records[0][1]
    start_node = []
    end_node = None
    next_node = None
    new_next_node = None
    
    for node in nodes:
        if node['att'] != None:
            if "STA" in node['att']:
                start_node += [node]
            if "END" in node['att']:
                end_node = node

    paths = []
    for node in start_node:
        total_template = []
        next_node = node
        new_next_node = node
        while(next_node != None):
            temp_node_graph = []
            for relation in relations:
                head = relation.nodes[0]
                tail = relation.nodes[1]
                relation_type = relation.type
                if head == next_node:
                    if relation_type != "Next_Operation":
                        temp_node_graph += [tail]
                    else:
                        new_next_node = tail
            text = graph_to_template(temp_node_graph, records, next_node)
            total_template += [text]
            
            if next_node != new_next_node:
                next_node = new_next_node
            else:
                is_end = True
                next_node = None 
        paths += ["".join([f"Step {x+1}:\n                    " + total_template[x] + "\n                " for x in range(len(total_template))])]
    output = start + "        Synthesis_path:\n            " + "".join([f"Path {x+1}:\n                " + paths[x] + "\n            " for x in range(len(paths))])
    #print(output)
    
    return output

def start_temp(records, driver, name:str):
    prc = prc_query(records, driver)
    ptp = ptp_query(records, driver)
    dev = dev_query(records, driver)

    prc_list = prc['ceid']
    ptp_list = ptp['ceid']
    dev_list = dev['ceid']

    prcs, ptps, devs = start_temp_query(prc_list, ptp_list, dev_list, records, driver)

    prc_temp = ""
    ptp_temp = ""
    dev_temp = ""
    if prc['prc'] != []:
        prc_temp = "\n            ". join(prcs)
    if dev['dev'] != []:
        dev_temp = "\n            ". join(devs)
    if ptp['ptp'] != []:
        ptp_temp = "\n            ". join(ptps)
    
    if prc_temp == "":
        prc_temp = "None."
    if dev_temp == "":
        dev_temp = "None."
    if ptp_temp == "":
        ptp_temp = "None."
    dev_temp = dev_temp.replace(" with amount", ": ")
    dev_temp = dev_temp.replace(" with number", ": ")
    dev_temp = dev_temp.replace("\n            H2O", "")
    dev_temp = dev_temp.replace("\n            O2", "")
    dev_temp = dev_temp.replace("\n            N2", "")
    dev_temp = dev_temp.replace(":  an", ": 1")
    dev_temp = dev_temp.replace(":  a", ": 1")
    start = f"""    In order to synthesis {name}, the following precursors, participants, and devices are used:
        Precursors:
            {prc_temp}
        Participants:
            {ptp_temp}
        Devices:
            {dev_temp}\n"""
    return start

def start_temp_query(prcs, ptps, devs, records, driver):
    
    nodes = records[0][0]
    new_prcs = []
    new_ptps = []
    new_devs = []
    for prc in prcs:
        for node in nodes:
            if prc == node['ceid']:
                new_prcs += [construct_nlp_from_graph(records, node)]

    for ptp in ptps:
        for node in nodes:
            if ptp == node['ceid']:
                new_ptps += [construct_nlp_from_graph(records, node)]

    for dev in devs:
        for node in nodes:
            if dev == node['ceid']:
                new_devs += [construct_nlp_from_graph(records, node)]

    return new_prcs, new_ptps, new_devs
        
    

def graph_to_template(graph, records, start_node):
    sop = start_node['name']
    prc = []
    dev = []
    cnd = []
    opt = []
    ptp = []
    for node in graph:
        if list(node.labels)[0] == 'PRC':
            prc += [construct_nlp_from_graph(records, node)]
        if list(node.labels)[0] == 'DEV':
            dev += [construct_nlp_from_graph(records, node)]
        if list(node.labels)[0] == 'CND':
            cnd += [construct_nlp_from_graph(records, node)]
        if list(node.labels)[0] == 'OPT':
            opt += [construct_nlp_from_graph(records, node)]
        if list(node.labels)[0] == 'PTP':
            ptp += [construct_nlp_from_graph(records, node)]

    prc_temp = "Precursors " + " and ".join(prc) + " are "
    dev_temp = " using " + " and ".join(dev)
    cnd_temp = " " + " ".join(cnd)
    opt_temp = ". This process generate " + " ".join(opt)
    ptp_temp = "and " + " ".join(ptp)
    
    if prc == []:
        prc_temp = "The generated materials from the last step are "
    if dev == []:
        dev_temp = ""
    if cnd == []:
        cnd_temp = ""
    if opt == []:
        opt_temp = ""
    if ptp == []:
        ptp_temp = ""
    
    template = f"""{prc_temp}{sop}{dev_temp}{ptp_temp}{cnd_temp}{opt_temp}."""
    
    return template

def construct_nlp_from_graph(records, start_node):
    heads_or_tail = None
    relation_type = []
    if 'PRC' in start_node.labels:
        heads_or_tail = 'tail'
        relation_type = ['Attribute_of']
    if 'DEV' in start_node.labels:
        heads_or_tail = 'tail'
        relation_type = ['Attribute_of', 'Amount_of', 'Number_of']
    if 'OPT' in start_node.labels:
        return start_node['name']
    if 'NUT' in start_node.labels:
        return start_node['name']
    if 'CND' in start_node.labels:
        return construct_nlp_from_graph_cnd(records, start_node)
    if 'PTP' in start_node.labels:
        return start_node['name']
    if 'NUM' in start_node.labels:
        heads_or_tail = 'tail'
        relation_type = ['Unit_of']
        
    nodes = records[0][0]
    relations = records[0][1]
    list_node = []
    
    for relation in relations:
        head = relation.nodes[0]
        tail = relation.nodes[1]
        relation_type_temp = relation.type
        if heads_or_tail == 'head' and relation_type_temp in relation_type:
            if head == start_node:
                list_node += [(relation_type_temp, tail)]
        if heads_or_tail == 'tail':
            if tail == start_node:
                if 'NUM' in start_node.labels and relation_type_temp == 'Number' and connect_to_sop(records, head) == False:
                    return construct_nlp_from_graph_cnd(records, head, special=True)
                elif relation_type_temp in relation_type:
                    list_node += [(relation_type_temp, head)]
           
    text = start_node['name']
    temp_text = []
    if list_node != []:
        for n in list_node:
            relation_type_temp, node_temp = n
            temp_text += [text_with_type(start_node, relation_type_temp) + construct_nlp_from_graph(records, node_temp)]
    
    return text + " ".join(temp_text)
        
def text_with_type(node, relation_type_temp):
    if 'PRC' in node.labels:
        return " with " + relation_type_temp[:-3].lower() + " "
    if 'DEV' in node.labels:
        return " with " + relation_type_temp[:-3].lower() + " "
    if 'NUT' in node.labels:
        return " "
    if 'NUM' in node.labels:
        return " "
    

def construct_nlp_from_graph_cnd(records, start_node, special=False):
    relations = records[0][1]
    conditions = []
    if special:
        num = ""
        nut = ""
        dev = ""
        temp_num = None
        for relation in relations:
            head = relation.nodes[0]
            tail = relation.nodes[1]
            relation_type_temp = relation.type
            if head == start_node:
                if list(tail.labels)[0] == 'DEV':
                    dev = tail['name']
                if list(tail.labels)[0] == 'NUM':
                    num = tail['name']
                    temp_num = tail
                    
        for relation in relations:
            head = relation.nodes[0]
            tail = relation.nodes[1]
            relation_type_temp = relation.type
            if tail == temp_num:
                if list(head.labels)[0] == 'NUT':
                    nut = head['name']

        return " ".join([dev, start_node['name'], num, nut])
        
    else:
        for relation in relations:
            head = relation.nodes[0]
            tail = relation.nodes[1]
            relation_type_temp = relation.type
            if head == start_node:
                conditions += [construct_nlp_from_graph(records, tail)]
        text = start_node['name'] + " " + " and ".join(conditions)
        return text


def connect_to_sop(records, start_node):
    nodes = records[0][0]
    relations = records[0][1]
    for relation in relations:
        head = relation.nodes[0]
        tail = relation.nodes[1]
        relation_type_temp = relation.type
        if tail == start_node and 'SOP' in head.labels:
            return True
    return False
    
def special_num(records, start_node):
    nodes = records[0][0]
    relations = records[0][1]
    for relation in relations:
        head = relation.nodes[0]
        tail = relation.nodes[1]
        relation_type_temp = relation.type
        if head == start_node and relation_type_temp == 'Attribute_of' and list(tail.labels)[0] == 'DEV':
            has_number = True
            

def main(unused_argv):
    driver = GraphDatabase.driver(host, auth=(user, password))
    text = graph_query("Li10SnP2S12", driver)
    #import pdb; pdb.set_trace()
    print(text)
    
    
    


if __name__ == "__main__":
    #add_options()
    app.run(main)




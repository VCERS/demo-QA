#!/usr/bin/python3

import re
import json
from os import listdir
from os.path import basename, splitext, isfile, join
from absl import app, flags
from neo4j import GraphDatabase
from typing import List

FLAGS = flags.FLAGS

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

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

def prc_query(records, driver):

    nodes = records[0][0]
    relation = records[0][1]
    set_of_precursor = list()
    ceid = list()
    for node in nodes:
        if "PRC" in node.labels:
            set_of_precursor.append(node['name'])
            ceid.append(node['ceid'])
        if "ELT" in node.labels:
            elt_name = node['name']
    results = dict(name = elt_name, prc=set_of_precursor, ceid=ceid)
    return results

def sop_query(records, driver):

    nodes = records[0][0]
    relation = records[0][1]
    set_of_sop = list()
    ceid = list()
    for node in nodes:
        if "SOP" in node.labels:
            set_of_sop.append(node['name'])
            ceid.append(node['ceid'])
        if "ELT" in node.labels:
            elt_name = node['name']
    results = dict(name = elt_name, sop=set_of_sop, ceid=ceid)
    return results

def ptp_query(records, driver):


    nodes = records[0][0]
    relation = records[0][1]
    set_of_ptp = list()
    ceid = list()
    for node in nodes:
        if "PTP" in node.labels:
            set_of_ptp.append(node['name'])
            ceid.append(node['ceid'])
        if "ELT" in node.labels:
            elt_name = node['name']
    results = dict(name = elt_name, ptp=set_of_ptp, ceid=ceid)
    return results

def dev_query(records, driver):

    nodes = records[0][0]
    relation = records[0][1]
    set_of_dev = list()
    ceid = list()
    for node in nodes:
        if "DEV" in node.labels:
            set_of_dev.append(node['name'])
            ceid.append(node['ceid'])
        if "ELT" in node.labels:
            elt_name = node['name']
    results = dict(name = elt_name, dev=set_of_dev, ceid=ceid)
    return results





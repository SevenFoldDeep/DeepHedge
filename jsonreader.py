# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:55:15 2019

@author: ahls_st
"""

import json


json_string = r'C:\Users\ahls_st\Documents\MasterThesis\4Steve\planet_order_348405\20190422_093532_1001\20190422_093532_1001_metadata.json'
datastore = json.loads(json_string)
print(datastore)
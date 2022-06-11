import json
import os

file_path = '../KM LEE/Study/JSON/test1.json'

with open(file_path, 'r') as fp:
    data = json.load(fp)
    
# print(data['Pie'])
# print(data['Pie']['gender'])

print(json.dumps(data, indent=4))

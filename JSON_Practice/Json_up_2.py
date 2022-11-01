import json
from textwrap import indent

with open('C:/KM LEE/JSON_Upgrade/states.json') as f:
    data = json.load(f)

for state in data['states']:
    del state['area_codes']
    
with open('C:/KM LEE/JSON_Upgrade/new_states.json', 'w') as f:
    json.dump(data, f, indent=2)
    # print(state['name'], state['abbreviation'])
# new_string = json.dumps(data, indent=2, sort_keys=True)
# print(new_string)
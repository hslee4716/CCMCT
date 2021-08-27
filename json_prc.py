import json

temp_dict = {
    "car": {"index": [1,2,3,4,5]},
    "horse" : 100
}

with open('test.json', 'w', encoding='utf-8') as make_file:
    json.dump(temp_dict, make_file, indent="\t")
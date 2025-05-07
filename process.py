import json

with open('data/database.json', 'r') as f:
    data = json.load(f)

res = []

for item in list(data.values())[:100]:
    res.append({"content": item["content"], "title": item["title"]})

with open('data/database_100.json', 'w') as f:
    json.dump(res, f, indent=4)

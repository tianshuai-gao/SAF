import json
import random

random.seed(0)

for lang in ['bo', 'ug', 'kk', 'mn']:
    with open(f'./{lang}/test.json') as f:
        data = json.load(f)
    data = random.sample(data, 200)
    with open(f'./{lang}/test.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
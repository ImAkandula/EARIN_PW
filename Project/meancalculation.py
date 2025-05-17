import json
from collections import defaultdict

data_file = r'C:\Users\adity\Desktop\EAIRN\EARIN_PW\Project\datasets\train_set.json'

category_word_counts = defaultdict(list)

with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        article = json.loads(line)
        category = article.get('category', 'Unknown')
        short_desc = article.get('short_description', '')
        word_count = len(short_desc.split())
        category_word_counts[category].append(word_count)

print("Category | Mean Short Description Word Count | Articles < Mean")
print("-------------------------------------------------------------")

total_less_than_mean = 0

for category, counts in category_word_counts.items():
    mean_size = sum(counts) / len(counts)
    less_than_mean_count = sum(1 for c in counts if c < mean_size)
    total_less_than_mean += less_than_mean_count
    print(f"{category} | {mean_size:.2f} | {less_than_mean_count}")

print("-------------------------------------------------------------")
print(f"Total articles with short descriptions less than their category mean: {total_less_than_mean}")

import json
import random
from collections import defaultdict

input_file = r'C:\Users\adity\Desktop\EAIRN\EARIN_PW\Project\datasets\Filtered_News_Category_Dataset.json'
sampled_file = r'C:\Users\adity\Desktop\EAIRN\EARIN_PW\Project\datasets\test_set.json'
remaining_file = r'C:\Users\adity\Desktop\EAIRN\EARIN_PW\Project\datasets\train_set.json'

# Read all articles and group by category
category_articles = defaultdict(list)

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        article = json.loads(line)
        category = article.get('category', 'Unknown')
        category_articles[category].append(article)

# Prepare output files
with open(sampled_file, 'w', encoding='utf-8') as f_sampled, \
     open(remaining_file, 'w', encoding='utf-8') as f_remaining:
    
    for category, articles in category_articles.items():
        n_total = len(articles)
        n_sample = max(1, int(n_total * 0.17))  # at least 1 article if available
        sampled = random.sample(articles, n_sample)
        remaining = [a for a in articles if a not in sampled]
        
        # Write sampled
        for art in sampled:
            f_sampled.write(json.dumps(art) + '\n')
        # Write remaining
        for art in remaining:
            f_remaining.write(json.dumps(art) + '\n')

print(f"Sampled 17% articles saved to: {sampled_file}")
print(f"Remaining 83% articles saved to: {remaining_file}")

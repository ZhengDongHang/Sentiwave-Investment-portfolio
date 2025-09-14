import pandas as pd

# 读取情感词汇文件
emotion_words_df = pd.read_csv("emo_dict/情感词汇.csv", encoding="utf-8", on_bad_lines='skip')

# 清除列名中的前后空格, 再打印列名以确认
emotion_words_df.columns = emotion_words_df.columns.str.strip()

# 创建两个空的字典，用来存储积极和消极词汇
positive_words = set()
negative_words = set()

# 根据情感分类将词汇分为积极和消极
for index, row in emotion_words_df.iterrows():

    # 获取情感分类（正面/负面/中性）
    sentiment_category = row['情感分类'].strip()

    # 根据情感分类将词汇添加到相应的集合
    if sentiment_category in ['PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK']:  # 积极类
        positive_words.add(row['词语'])
    elif sentiment_category in ['NA', 'NB', 'NJ', 'NH', 'PF', 'NI', 'NC', 'NG', 'ND', 'NN', 'NK', 'NL']:  # 消极类
        negative_words.add(row['词语'])

# 将积极和消极词汇保存到文件
with open('emo_dict/positive_words.txt', 'w', encoding='utf-8') as f:
    for word in positive_words:
        f.write(word + '\n')

with open('emo_dict/negative_words.txt', 'w', encoding='utf-8') as f:
    for word in negative_words:
        f.write(word + '\n')

print("情感词典预处理完成！")

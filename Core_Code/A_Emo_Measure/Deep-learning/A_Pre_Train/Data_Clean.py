def clean_dataset(data_list):
    """
    清洗 guba_data 的字典列表：
    - 删除长度小于 5 的文本
    - 删除纯数字文本

    参数：
    data_list (list of dict): 原始数据列表，每个元素包含 'guba_data'

    返回：
    list of dict: 清洗后的数据列表
    """


    def is_valid(text):
        try:
            text = text.strip()
            return len(text) >= 5 and not text.isdigit()
        except:
            return False

    cleaned_data = []
    for item in data_list:
        cleaned_item = item.copy()
        cleaned_item['guba_data'] = [text for text in item.get('guba_data', []) if is_valid(text)]
        cleaned_data.append(cleaned_item)

    return cleaned_data
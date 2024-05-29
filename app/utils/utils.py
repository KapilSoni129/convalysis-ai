import re

def extract_json(string):
    start_idx = 0
    end_idx = len(string)
    if re.search('json\n', string):
        start_idx = string.find('json\n') + len('json\n')
        end_idx = string.rfind('```')   
    elif re.search('JSON\n', string):
        start_idx = string.find('JSON\n') + len('JSON\n')
        end_idx = string.rfind('```')
    elif re.search('```\n', string):
        start_idx = string.find('```') + len('```')
        end_idx = string.rfind('```')
    return string[start_idx:end_idx]
from prettytable import PrettyTable
from typing import List, Text, Dict, Union

def convert_size(num, suffix='B'):
    for unit in ['', ' Ki', ' Mi', ' Gi', ' Ti', ' Pi', ' Ei', ' Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def create_list_item_table(items: List[Dict]):
    from columnar import columnar
    headers = ["Model name", "Size", "Created at", "Link"]
    data = []
    for item in items:
        data.append([item["key"], convert_size(item["size"]), item["created_at"][:10], item["url"]])
    table = columnar(data, headers, no_borders=False)
    return table

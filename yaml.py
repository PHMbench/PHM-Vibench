import json

def safe_load(stream):
    if hasattr(stream, 'read'):
        return json.load(stream)
    return json.loads(stream)


def safe_dump(data, stream=None, allow_unicode=True):
    dumped = json.dumps(data, ensure_ascii=not allow_unicode)
    if stream is None:
        return dumped
    stream.write(dumped)

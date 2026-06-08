def parse_kv_string(s):
    """Parse 'key=value,key=value' into a dict.

    Whitespace around keys and values is stripped.
    Empty input returns an empty dict.
    Pairs without '=' raise ValueError.
    """
    if not s or not s.strip():
        return {}
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(f"malformed pair: {pair!r}")
        k, _, v = pair.partition("=")
        out[k.strip()] = v.strip()
    return out

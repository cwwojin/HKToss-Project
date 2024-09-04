def parse_boolean(bool_str: str):
    try:
        return bool_str.lower() == "true"
    except:
        return False

def postprocess(candidates):
    x = []
    for candidate in candidates:
        candidate.replace('\n', ' ')
        if candidate[-1]=='.':
            x.append(candidate)
        elif candidate.find('.') != -1:
            clast_comma = candidate[:-(candidate[::-1].find('.'))]
            x.append(clast_comma)
    x = list(set(x))
    if x:
        return x
    return candidates
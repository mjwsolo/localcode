def title(s):
    return s.lower()


def initials(name):
    return ''.join(p[0] for p in name.split(' '))

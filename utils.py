EPSILON = 0.0001
MATRIX_EPSILON = 0.01
MAX_CHARACTER_LENGTH = 70


def auto_str(cls):
    def __str__(self):
        return '(%s)' % (', '.join('%s=%s' % item for item in vars(self).items()))

    cls.__str__ = __str__
    return cls

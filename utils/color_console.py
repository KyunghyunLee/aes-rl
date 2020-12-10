
"""
Colored console

Originated from:
https://github.com/apourchot/CEM-RL/blob/master/util.py

"""
from colour import Color
from colored import fg, bg, attr


def prRed(prt):
    return "\033[91m{}\033[00m" .format(prt)


def prGreen(prt):
    return "\033[92m{}\033[00m" .format(prt)


def prYellow(prt):
    return "\033[93m{}\033[00m" .format(prt)


def prLightPurple(prt):
    return "\033[94m{}\033[00m" .format(prt)


def prPurple(prt):
    return "\033[95m{}\033[00m" .format(prt)


def prCyan(prt):
    return "\033[96m{}\033[00m" .format(prt)


def prLightGray(prt):
    return "\033[97m{}\033[00m" .format(prt)


def prBlack(prt):
    return "\033[98m{}\033[00m" .format(prt)


def prAuto(prt):
    if '[INFO]' in prt:
        return prGreen(prt)
    elif '[WARNING]' in prt:
        return prYellow(prt)
    elif '[ERROR]' in prt:
        return prRed(prt)
    return prt


def prValuedColor(value, low, high, n, color_from, color_to):
    assert low < high
    c_from = Color(color_from)
    c_to = Color(color_to)
    colors = list(c_from.range_to(c_to, n))
    value_range = high - low
    one_value = float(value_range) / n
    if value < low:
        value = low
    elif value > high:
        value = high

    value_cat = int(round((value - low) / one_value))

    if value_cat >= len(colors):
        value_cat = len(colors) - 1

    color_code = colors[value_cat].get_hex_l()

    # return prColor(str(value), fore=color_code)
    return color_code


def prColor(prt, fore=None, back=None, res=True):
    if fore is None and back is None:
        return prt

    code = ''
    res_code = ''
    if res:
        res_code = attr(0)

    if fore is not None:
        code += fg(fore)

    if back is not None:
        code += bg(back)

    return code + prt + res_code

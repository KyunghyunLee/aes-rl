from .color_console import prAuto
from typing import Union


class AndStatement:
    def __init__(self, cond_list: Union[list, tuple]):
        assert len(cond_list) > 1
        self.cond_list = cond_list

    def __call__(self, args_dict):
        ret_flag = True
        for cond in self.cond_list:
            ret = cond(args_dict)
            ret_flag = ret_flag and ret
        return ret_flag

    def __str__(self):
        str1 = ''
        for idx, cond in enumerate(self.cond_list):
            if idx == 0:
                str1 += f'({cond})'
            else:
                str1 += f' And ({cond})'
        return str1

    def cur_value(self, args_dict):
        str1 = ''
        for idx, cond in enumerate(self.cond_list):
            if idx == 0:
                str1 += f'({cond.cur_value(args_dict)})'
            else:
                str1 += f', ({cond.cur_value(args_dict)})'
        return str1


class OrStatement:
    def __init__(self, cond_list: Union[list, tuple]):
        assert len(cond_list) > 1
        self.cond_list = cond_list

    def __call__(self, args_dict):
        ret_flag = False
        for cond in self.cond_list:
            ret = cond(args_dict)
            ret_flag = ret_flag or ret
        return ret_flag

    def __str__(self):
        str1 = ''
        for idx, cond in enumerate(self.cond_list):
            if idx == 0:
                str1 += f'({cond})'
            else:
                str1 += f' Or ({cond})'
        return str1

    def cur_value(self, args_dict):
        str1 = ''
        for idx, cond in enumerate(self.cond_list):
            if idx == 0:
                str1 += f'({cond.cur_value(args_dict)})'
            else:
                str1 += f', ({cond.cur_value(args_dict)})'
        return str1


class ArgStatement:
    """
    Argument Statement
    arg_key: key in args
    cond:
        'equal': arg == value
        'not equal': arg != value
        'in': arg in value
        'not in': arg not in value
        'greater': arg > value
        'greater equal': arg >= value
        'less': arg < value
        'less equal': arg <= value
    """
    def __init__(self, arg_key, cond, value):
        self.arg_key = arg_key
        self.cond = cond
        self.value = value

        conditions = ['equal', 'not equal', 'in', 'not in', 'greater', 'greater equal', 'less', 'less equal']
        assert self.cond in conditions

    def __call__(self, args_dict):
        cond = self.cond
        key = self.arg_key
        value = self.value

        if cond == 'equal':
            if args_dict[key] == value:
                return True
        elif cond == 'not equal':
            if args_dict[key] != value:
                return True
        elif cond == 'in':
            if args_dict[key] in value:
                return True
        elif cond == 'not in':
            if args_dict[key] not in value:
                return True
        elif cond == 'greater':
            if args_dict[key] > value:
                return True
        elif cond == 'greater equal':
            if args_dict[key] >= value:
                return True
        elif cond == 'less':
            if args_dict[key] < value:
                return True
        elif cond == 'less equal':
            if args_dict[key] <= value:
                return True

        return False

    def __str__(self):
        cond_text = {
            'equal': '==',
            'not equal': '!=',
            'in': 'in',
            'not in': 'not in',
            'greater': '>',
            'greater equal': '>=',
            'less': '<',
            'less equal': '<='
        }
        return '{} {} {}'.format(self.arg_key, cond_text[self.cond], self.value)

    def cur_value(self, args_dict):
        return '{} = {}'.format(self.arg_key, args_dict[self.arg_key])


class TrueStatement:
    def __call__(self, args_dict):
        return True

    def cur_value(self, args_dict):
        return 'External True'


class FalseStatement:
    def __call__(self, args_dict):
        return False

    def cur_value(self, args_dict):
        return 'External False'


class ArgCondition:
    """
    Argument condition definition
    """

    def __init__(self, cond_statement: Union[ArgStatement, AndStatement, OrStatement, bool],
                 true_statement: Union[ArgStatement, AndStatement, OrStatement, bool]=None,
                 false_statement: Union[ArgStatement, AndStatement, OrStatement,  bool]=None):
        if isinstance(cond_statement, bool):
            self.cond_statement = TrueStatement() if cond_statement else FalseStatement()
        else:
            self.cond_statement = cond_statement

        self.true_statement = None
        if true_statement is not None:
            if isinstance(true_statement, bool):
                self.true_statement = TrueStatement() if true_statement else FalseStatement()
            else:
                self.true_statement = true_statement

        self.false_statement = None
        if false_statement is not None:
            if isinstance(false_statement, bool):
                self.false_statement = TrueStatement() if false_statement else FalseStatement()
            else:
                self.false_statement = false_statement

    def __call__(self, arg_dict, verbose=True):
        if self.cond_statement(arg_dict):
            if self.true_statement is not None:
                if self.true_statement(arg_dict):
                    return True
                else:
                    if verbose:
                        print(prAuto(f'[ERROR] If ({self.cond_statement}) then ({self.true_statement}). '
                                     f'Currently, ({self.true_statement.cur_value(arg_dict)})'))
                    return False
            return True
        else:
            if self.false_statement is not None:
                if self.false_statement(arg_dict):
                    return True
                else:
                    if verbose:
                        print(prAuto(f'[ERROR] If not ({self.cond_statement}) then ({self.false_statement}). '
                                     f'Currently, ({self.false_statement.cur_value(arg_dict)})'))
                    return False
            else:
                if self.true_statement is None:
                    if verbose:
                        print(prAuto(f'[ERROR] ({self.cond_statement}) not True. '
                                     f'Currently, ({self.cond_statement.cur_value(arg_dict)})'))
                    return False
                return True


class ArgCheck:
    def __init__(self):
        self.conditions = []

    def add_condition(self, condition: ArgCondition):
        self.conditions.append(condition)

    def check(self, args):
        ok_flag = True
        arg_dict = vars(args)
        for cond in self.conditions:
            if not cond(arg_dict):
                ok_flag = False

        return ok_flag




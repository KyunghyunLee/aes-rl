import builtins
import os
import git

from .color_console import prAuto


class Printer(object):
    """
    Simple printer
    """
    def __init__(self):
        # print(prAuto('[WARNING] Logging disabled'))
        pass

    def log(self, txt, end='\n'):
        print(txt, end=end)


class Logger(Printer):
    """
    Logger
    """
    def __init__(self, path, name='log.txt'):
        # super(Logger, self).__init__()
        self.path = path
        self.name = name
        self.full_path = os.path.join(self.path, name)
        os.makedirs(self.path,  exist_ok=True)
        gitdir = os.getcwd()
        repo = git.Repo(gitdir)
        gittag = str(repo.head.object.hexsha)[:6]
        str1 = f'Git Tag: {gittag}, is_dirty: {repo.is_dirty()}'

        with open(self.full_path, 'w') as f:
            f.write(str1 + '\n')
        print(str1)

    def log(self, txt, end='\n', logonly=False):
        if os.path.exists(self.full_path):
            with open(self.full_path, 'a') as f:
                f.write(txt)
                f.write(end)

        if not logonly:
            print(txt, end=end)

import datetime
import os


class Timestamp(object):
    def __init__(self, base_dir, enable=True):
        self.file = os.path.join(base_dir, 'timestamp.txt')
        self.stamp_dict = {}
        time1 = datetime.datetime.now()
        time_str = time1.strftime('%y-%m-%d %H:%M:%S:%f')
        self.indent = 0
        self.enable = enable
        if enable:
            with open(self.file, "w") as f:
                f.write(time_str)
                f.write('\n')

    def tic(self, name, now1=None):
        if not self.enable:
            return

        if now1 is None:
            now1 = datetime.datetime.now()
        if isinstance(name, list):
            name_list = name
            for name in name_list:
                self.tic(name, now1)
        else:
            if name in self.stamp_dict:
                self.stamp_dict[name].append(now1)
            else:
                self.stamp_dict[name] = [now1]
            with open(self.file, 'a') as f:
                f.write('{}{}] Tic\n'.format('  ' * self.indent, name))
            self.indent += 1

    def toc(self, name, now1=None):
        if not self.enable:
            return

        if now1 is None:
            now1 = datetime.datetime.now()
        if isinstance(name, list):
            name_list = name
            for name in name_list:
                self.toc(name, now1)
        else:
            if name not in self.stamp_dict:
                return

            self.indent -= 1

            with open(self.file, 'a') as f:
                if len(self.stamp_dict[name]) != 1:
                    for idx, item in enumerate(self.stamp_dict[name]):
                        f.write('{}{}_{}] Toc {:.6f}\n'.format('  '*self.indent, name, idx, (now1 - item).total_seconds()))
                else:
                    f.write('{}{}] Toc {:.6f}\n'.format('  '*self.indent, name, (now1 - self.stamp_dict[name][0]).total_seconds()))

            del self.stamp_dict[name]

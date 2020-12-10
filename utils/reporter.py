import ray
import copy
import datetime
import time


@ray.remote(num_cpus=0.5, resources={'head': 0.01})
class TrainReporter:
    def __init__(self, n):
        self.n = n
        self.status = [{} for _ in range(n)]

    def report(self):
        pass

    def update(self, individual_idx, key, value):
        self.status[individual_idx][key] = value

    def get_status(self):
        return self.status


@ray.remote(num_cpus=0.1, resources={'head': 0.01})
class TrainReporterHelper:
    def __init__(self, reporter, interval=0.1):
        self.interval = interval
        self.reporter = reporter

    def run(self):
        prev_run = datetime.datetime.now()
        while True:
            now1 = datetime.datetime.now()
            if (now1 - prev_run).total_seconds() >= self.interval:
                prev_run = now1
                self.reporter.report.remote()


@ray.remote(num_cpus=0.1, resources={'head': 0.01})
class TrainReporterClock:
    def run(self, second):
        time.sleep(second)

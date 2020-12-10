import ray
from utils.util import *
from utils.logger import Printer, Logger


class Manager(object):
    def __init__(self, logger=None):
        self.workers = {}
        self.jobs = {}
        self.logger = logger if logger is not None else Printer()

    def add_worker(self, worker, name=None):
        if name is None:
            name = '__NONAME__'
        if name not in self.workers:
            self.workers[name] = []
            self.jobs[name] = []

        new_worker = {
            'worker': worker,
            'status': 'idle',
            'func_name': '',
            'job_name': None,
            'job': None,
            'args': None,
            'kwargs': None,
            'setting': None,
            'returned': 0
        }
        self.workers[name].append(new_worker)

    def num_running_worker(self, name=None):
        if name is None:
            name = '__NONAME__'

        if isinstance(name, list):
            len_running_workers = 0
            for one_name in name:
                len_running_workers += self.num_running_worker(one_name)
            return len_running_workers

        running_workers = [worker for worker in self.workers[name] if worker['status'] == 'working']
        return len(running_workers)

    def num_idle_worker(self, name=None):
        if name is None:
            name = '__NONAME__'

        idle_workers = [worker for worker in self.workers[name] if worker['status'] == 'idle']
        return len(idle_workers)

    def get_index(self, worker, name=None):
        if name is None:
            name = '__NONAME__'

        for idx, worker1 in enumerate(self.workers[name]):
            if worker == worker1['worker']:
                return idx

        return None

    def get_idle_worker(self, name=None, idx=0):
        if name is None:
            name = '__NONAME__'
        idle_workers = [worker for worker in self.workers[name] if worker['status'] == 'idle']
        if len(idle_workers) < idx:
            return None
        worker = idle_workers[idx]['worker']

        return worker, self.get_index(worker, name)

    def get_worker_state_by_index(self, name=None, idx=0):
        if name is None:
            name = '__NONAME__'
        workers = [worker for worker in self.workers[name]]
        if len(workers) < idx:
            return None

        return workers[idx]['status']

    def get_worker_by_index(self, name=None, idx=0):
        if name is None:
            name = '__NONAME__'
        workers = [worker for worker in self.workers[name]]
        if len(workers) < idx:
            return None
        worker = workers[idx]['worker']
        return worker

    def new_job(self, func_name, *args, specific_worker=None, job_name=None, job_setting=None, **kwargs):
        if job_name is None:
            job_name = '__NONAME__'

        if job_name not in self.workers:
            self.logger.log(prAuto('[ERROR] No worker named {}'.format(job_name)))
            raise KeyError

        idle_workers = [worker for worker in self.workers[job_name] if worker['status'] == 'idle']
        if len(idle_workers) == 0:
            self.logger.log(prAuto('[ERROR] All workers are busy'))
            raise ValueError('[ERROR] All workers are busy')

        if specific_worker is not None:
            worker = None
            for idx in range(len(idle_workers)):
                if idle_workers[idx]['worker'] == specific_worker:
                    worker = idle_workers[idx]
                    break
            if worker is None:
                raise ValueError
        else:
            worker = idle_workers[0]

        func = getattr(worker['worker'], func_name, None)
        if func is None:
            self.logger.log(prAuto('[ERROR] No function named {} in worker {}'.format(func_name, job_name)))
            raise ValueError('[ERROR] No function named {} in worker {}'.format(func_name, job_name))

        worker['status'] = 'working'
        worker['job_name'] = job_name
        worker['func_name'] = func_name
        worker['args'] = args
        worker['kwargs'] = kwargs
        worker['setting'] = job_setting
        worker['returned'] = 0
        worker['job'] = func.remote(*args, **kwargs)
        self.jobs[job_name].append(worker['job'])

    def wait(self, name=None, timeout=None, remove=True):
        if name is None:
            name = self.workers.keys()

        if isinstance(name, str):
            name = [name]

        all_jobs = []

        for one_name in name:
            all_jobs += self.jobs[one_name]

        jobs_done, jobs_running = ray.wait(all_jobs, timeout=timeout)

        if len(jobs_done) == 0:
            waiting_worker = None

            for one_name in name:
                for worker in self.workers[one_name]:
                    if worker['status'] == 'waiting':
                        waiting_worker = worker
                        break
            if waiting_worker is not None:
                waiting_worker['returned'] += 1
                return waiting_worker['job_name'], waiting_worker['job'], waiting_worker

            return None

        finished_job = None
        finished_worker = None

        job = jobs_done[0]
        job_name = None

        for one_name in name:
            if job in self.jobs[one_name]:
                finished_job = one_name
                for worker in self.workers[one_name]:
                    if worker['job'] == job:
                        finished_worker = worker
                        break

                if finished_worker is not None:
                    job_name = one_name
                    break

        if remove:
            finished_worker['status'] = 'idle'
            self.jobs[job_name].remove(job)
        else:
            finished_worker['status'] = 'waiting'
        finished_worker['returned'] += 1
        return finished_job, job, finished_worker

    def done(self, worker):
        worker['status'] = 'idle'
        self.jobs[worker['job_name']].remove(worker['job'])
        worker['job_name'] = None
        worker['job'] = None
        worker['returned'] = 0

    def kill_all(self):
        for one_name in self.workers:
            for worker in self.workers[one_name]:
                ray.kill(worker['worker'])



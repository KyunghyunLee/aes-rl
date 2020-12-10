import ray
import git
import os
import argparse
import torch
import GPUtil
import os


class GitWorker(object):
    def __init__(self):
        pass

    def check(self, sha):
        try:
            gitdir = os.path.dirname(os.path.abspath(__file__))
            repo = git.Repo(gitdir)
        except:
            gitdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
            repo = git.Repo(gitdir)

        if repo.is_dirty():
            print('False')
            return False

        my_sha = repo.head.object.hexsha
        if my_sha != sha:
            print('False')
            return False

        print('True')
        return True

    def sync(self, sha):
        try:
            gitdir = os.path.dirname(os.path.abspath(__file__))
            repo = git.Repo(gitdir)
        except:
            gitdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
            repo = git.Repo(gitdir)

        my_sha = repo.head.object.hexsha
        if my_sha != sha:
            g = git.cmd.Git(gitdir)
            g.pull('origin', 'master')

        print('Git Tag: {}'.format(repo.head.object.hexsha))
        return True

    def get_gpu_count(self):
        return len(GPUtil.getGPUs())

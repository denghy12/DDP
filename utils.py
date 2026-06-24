import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
import torch.optim as optim
import pickle
import numpy as np
import os
import os.path as osp

def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)
    
class ReDirectSTD(object):

  def __init__(self, fpath=None, console='stdout', immediately_visible=False):
    import sys
    import os
    import os.path as osp

    assert console in ['stdout', 'stderr']
    self.console = sys.stdout if console == 'stdout' else sys.stderr
    self.file = fpath
    self.f = None
    self.immediately_visible = immediately_visible
    if fpath is not None:
      # Remove existing log file.
      if osp.exists(fpath):
        os.remove(fpath)

    # Overwrite
    if console == 'stdout':
      sys.stdout = self
    else:
      sys.stderr = self

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      may_make_dir(os.path.dirname(osp.abspath(self.file)))
      if self.immediately_visible:
        with open(self.file, 'a') as f:
          f.write(msg)
      else:
        if self.f is None:
          self.f = open(self.file, 'w')
        self.f.write(msg)

  def flush(self):
    self.console.flush()
    if self.f is not None:
      self.f.flush()
      import os
      os.fsync(self.f.fileno())

  def close(self):
    self.console.close()
    if self.f is not None:
      self.f.close()
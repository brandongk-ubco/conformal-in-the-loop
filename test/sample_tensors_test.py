import numpy as np
import torch
import os 

from citl.utils import sample_tensors

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

class TestSampleTensors:

    def test_sample_empty_tensors(self):
        t1 = torch.tensor([])
        t2 = torch.tensor([])
        t1_, t2_ = sample_tensors(t1, t2)
        assert len(t1_) == 0
        assert len(t2_) == 0

    def test_sample_same_tensors(self):
        t1 = torch.tensor([1, 2, 3, 4])
        t2 = torch.tensor([1, 2, 3, 4])
        t1_, t2_ = sample_tensors(t1, t2, percentage=0.5)
        assert len(t1_) == 2
        assert len(t2_) == 2
        assert torch.equal(t1_, t2_)
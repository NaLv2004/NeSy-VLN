"""Executable boundary and reverse-proof checks, independent of Torch."""
import numpy as np
from hard_interpreter import Program, plan, corrected_state


def main():
    transitions=np.tile(np.arange(512,dtype=np.int16)[:,None],(1,17))
    correction=np.tile(np.arange(512,dtype=np.int16)[None],(512,1))
    transitions[1,4]=3;transitions[2,6]=3
    confidence=np.full((512,17),.1,np.float32)
    confidence[1,4]=.9;confidence[2,6]=.9;confidence[3,0]=.99
    program=Program(transitions,confidence,correction,3,3).validate()
    left=plan(program,9,1,horizon=2);right=plan(program,9,2,horizon=2)
    assert left['found'] and right['found']
    assert left['actions']==[4,0] and right['actions']==[6,0]
    assert left['states']==[1,3,3] and right['states']==[2,3,3]
    assert not plan(program,9,7,horizon=2)['found']
    for invalid in (np.array([1]),np.ones((49,)),np.ones((224,224,3)),1.0):
        try:corrected_state(program,9,invalid)
        except TypeError:pass
        else:raise AssertionError('Non-ID observation accepted')
    print('PASS: observation-conditioned hard lookup, complete reverse proofs, unreachable failure, scalar-only boundary, arbitrary renewal step')


if __name__=='__main__':main()

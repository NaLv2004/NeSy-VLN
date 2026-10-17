"""Immutable current curriculum; historical datasets are not overwritten."""
import json,hashlib
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
PROTOCOL=ROOT/'normal_near_medium_v67'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protocol():
    m=read(PROTOCOL/'manifest.json')
    assert sha(PROTOCOL/'train_tasks.json')==m['train_sha256']
    assert sha(PROTOCOL/'development80.json')==m['development_sha256']
    return m
def make_bank():
    from predictive_nesy.environment import make_bank as original
    bank=original();m=protocol()
    bank.train=read(PROTOCOL/'train_tasks.json');bank.dev=read(PROTOCOL/'development80.json')
    assert len(bank.dev)==80 and all(e['domain']=='normal' and e['distance_group'] in ('near','medium') for e in bank.train+bank.dev)
    assert not {e['scene'] for e in bank.train}&{e['scene'] for e in bank.dev}
    bank.groups={}
    for e in bank.train:bank.groups.setdefault((e['domain'],e['category'],e['distance_group']),[]).append(e)
    bank.provenance.update(train_layouts=m['train_layouts'],development_layouts=m['development_layouts'],curriculum=m)
    return bank

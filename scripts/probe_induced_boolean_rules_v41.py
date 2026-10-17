"""Learn explicit conjunctive rules from frozen discovered Boolean tokens.

Diagnostic only. No pixels, semantic predicate labels, or original evaluation
tasks are loaded. Every action probability is replayed from leaf rule proofs.
"""
import collections
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

base = Path('/home/user/xjy/nesy_vla_20261010')
source = base/'frozen_rule_learning_v39/symbols.json'
data = json.loads(source.read_text())
out = base/'induced_boolean_rules_v41'; out.mkdir(exist_ok=True)
lock = (out/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
assert not previous or previous['state']!='complete', 'Completed diagnostic must not be repeated'

def save(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)

result = {'state':'running','time':time.time(),'pid':os.getpid(),
          'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'training_inputs':'Automatically learned hard Boolean predicate tokens and task category only',
          'original_evaluation_data_used':False,'navigation_success_evaluated':False,
          'limits':'Single-frame movement-only imitation. Different expert actions can be equally valid. '
          'This is a frozen-representation capacity study, not a new navigation checkpoint.', 'arms':{}}
if previous:
    assert previous['source_sha256']==result['source_sha256']
    result['arms']=previous['arms']
    result['resume']={'previous_pid':previous['pid'],'completed_arms_preserved':list(previous['arms']),
                      'repair':'Randomized split thresholds anywhere strictly between zero and one encode the same Boolean literal'}
save(out/'result.json',result)

def extract(tree, classes):
    rules=[]
    def visit(node, positive, negative):
        feature=int(tree.tree_.feature[node])
        if feature<0:
            value=tree.tree_.value[node,0].astype(float); value/=value.sum()
            conclusion=np.zeros(17); conclusion[np.asarray(classes,dtype=int)]=value
            rules.append({'true':positive,'false':negative,'conclusion':conclusion.tolist(),
                          'training_support':int(tree.tree_.n_node_samples[node])})
            return
        assert 0 <= tree.tree_.threshold[node] < 1
        visit(tree.tree_.children_left[node],positive,negative+[feature])
        visit(tree.tree_.children_right[node],positive+[feature],negative)
    visit(0,[],[])
    return rules

def replay(forest, X):
    total=np.zeros((len(X),17))
    for rules in forest:
        count=np.zeros(len(X),dtype=int)
        for rule in rules:
            assert not(set(rule['true']) & set(rule['false']))
            active=(X[:,rule['true']]==1).all(1)&(X[:,rule['false']]==0).all(1)
            count+=active
            total+=active[:,None]*np.array(rule['conclusion'])[None]
        assert (count==1).all(), 'A decision tree must prove exactly one leaf per discrete state'
    return total/len(forest)

def measure(rows,p):
    labels=np.array([r['label'] for r in rows]); choices=p.argmax(1)
    outcomes=np.array([r['outcomes'] for r in rows]); moves=choices>0
    turns=~np.isin(labels,[0,4,11])
    return {'count':len(rows),'expert_correct':int((choices==labels).sum()),
            'expert_accuracy':float((choices==labels).mean()),
            'turn_correct':int(((choices==labels)&turns).sum()),'turn_count':int(turns.sum()),
            'collisions':int(outcomes[np.arange(len(rows))[moves],choices[moves]-1,0].sum()),
            'no_motion':int(outcomes[np.arange(len(rows))[moves],choices[moves]-1,1].sum()),
            'choices':dict(collections.Counter(map(int,choices))),
            'nll':float(-np.log(p[np.arange(len(rows)),labels].clip(1e-9)).mean())}

factories={
    'tree_depth6_leaf5':lambda:DecisionTreeClassifier(max_depth=6,min_samples_leaf=5,random_state=4100),
    'tree_depth12_leaf2':lambda:DecisionTreeClassifier(max_depth=12,min_samples_leaf=2,random_state=4100),
    'forest_depth12_leaf2':lambda:RandomForestClassifier(n_estimators=64,max_depth=12,min_samples_leaf=2,max_features=None,n_jobs=2,random_state=4100),
    'extra_depth12_leaf2':lambda:ExtraTreesClassifier(n_estimators=64,max_depth=12,min_samples_leaf=2,max_features=None,n_jobs=2,random_state=4100),
}
for name,factory in factories.items():
    if name in result['arms']:continue
    arm={}; combined={'train':[],'holdout':[]}; predicted={'train':[],'holdout':[]}; all_rules={}
    for category in ['chair','table']:
        subsets={s:[r for r in data['examples'] if r['category']==category and r['heldout']==(s=='holdout')]
                 for s in combined}
        X=np.array([r['codes'] for r in subsets['train']]); y=np.array([r['label'] for r in subsets['train']])
        learner=factory().fit(X,y)
        forest=[extract(t,learner.classes_) for t in getattr(learner,'estimators_',[learner])]
        all_rules[category]=forest
        checks={}
        for split,rows in subsets.items():
            X=np.array([r['codes'] for r in rows]); p=replay(forest,X)
            expected=np.zeros_like(p);expected[:,learner.classes_.astype(int)]=learner.predict_proba(X)
            assert np.allclose(p,expected,atol=1e-12)
            checks[split]=measure(rows,p); combined[split].extend(rows); predicted[split].append(p)
        arm[category]={'metrics':checks,'trees':len(forest),'rule_count':sum(map(len,forest)),
                       'max_premise_length':max(len(r['true'])+len(r['false']) for t in forest for r in t)}
    arm['overall']={s:measure(combined[s],np.concatenate(predicted[s])) for s in combined}
    save(out/(name+'_rules.json'),all_rules)
    result['arms'][name]=arm;result['time']=time.time();save(out/'result.json',result)
    print(json.dumps({'arm':name,**arm['overall']}),flush=True)
result.update(state='complete',time=time.time());save(out/'result.json',result)

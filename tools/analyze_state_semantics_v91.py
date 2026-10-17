"""Descriptive per-symbol tables and leave-one-layout-out semantic decoders."""
import csv,html,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,mutual_info_score
ROOT=Path(__file__).resolve().parents[1]/'output/state_semantics_v91';ROOT.mkdir(exist_ok=True)
data=json.loads((ROOT/'measurements.json').read_text());raw=data['rows']
seen=set();rows=[]
for r in raw:
 key=(r['category'],r['scene'],r['image_sha256'])
 if key not in seen:rows.append(r);seen.add(key)
for r in rows:
 for side in ['left','center','right']:r['blocked_'+side]=str(r['obstacles'][side]['collision'])
 r['target_visible']=str(r['target_visible']);r['eligible']=str(r['eligible'])
 r['action_id']=str(r['action'])
 for f in range(4):r['belief_peak_'+str(f)]=max(r['belief'][f])
fields=['target_visible','target_image_side','target_bearing_bin','distance_bin','eligible','blocked_left','blocked_center','blocked_right','action_group','action_id']
categories=['chair','table'];descriptors=[];usage=[]
def entropy(vals):
 c=np.asarray(list(Counter(vals).values()),float);p=c/c.sum();return float(-(p*np.log(p)).sum())
for category in categories:
 pool=[r for r in rows if r['category']==category]
 for kind in ['observed','memory']:
  for f in range(4):
   counts=Counter(r[kind][f] for r in pool)
   usage.append(dict(category=category,kind=kind,factor=f,active=len(counts),effective=math.exp(entropy([r[kind][f] for r in pool])),counts=dict(counts),
    mean_belief_peak=float(np.mean([r['belief_peak_'+str(f)] for r in pool])) if kind=='memory' else None))
   for symbol in range(16):
    subset=[r for r in pool if r[kind][f]==symbol];n=len(subset)
    row=dict(category=category,kind=kind,factor=f,symbol=symbol,count=n,episodes=len(set(r['episode_id'] for r in subset)),scenes=len(set(r['scene'] for r in subset)))
    relations=[]
    for field in fields:
     values=Counter(r[field] for r in subset);base=Counter(r[field] for r in pool)
     row[field]=dict(values)
     if not n:continue
     label,hits=values.most_common(1)[0];rate=hits/n;base_rate=base[label]/len(pool)
     scene_rates={scene:sum(r[field]==label for r in subset if r['scene']==scene)/sum(r['scene']==scene for r in subset) for scene in sorted(set(r['scene'] for r in subset))}
     relations.append(dict(field=field,label=label,fraction=rate,baseline=base_rate,lift=rate/base_rate,scene_rates=scene_rates))
    row['relations']=sorted(relations,key=lambda x: (x['fraction']-x['baseline']),reverse=True)
    if n:
     row['mean_distance']=float(np.mean([r['distance'] for r in subset]));row['median_distance']=float(np.median([r['distance'] for r in subset]))
     row['collision_rate']=float(np.mean([r['collision'] for r in subset]));row['mean_stop_probability']=float(np.mean([r['probabilities'][0] for r in subset]))
     row['mean_probabilities']=np.mean([r['probabilities'] for r in subset],axis=0).tolist()
     row['examples']=[];used_episodes=set()
     for r in subset:
      if r['episode_id'] not in used_episodes:row['examples'].append(r['frame']);used_episodes.add(r['episode_id'])
      if len(row['examples'])>=6:break
    descriptors.append(row)

features={
 'observed4':lambda r:np.eye(16)[r['observed']].flatten(),
 'memory_argmax4':lambda r:np.eye(16)[r['memory']].flatten(),
 'memory_soft64':lambda r:np.asarray(r['belief']).flatten(),
}
for kind in ['observed','memory']:
 for f in range(4):features[f'{kind}_factor{f}']=lambda r,kind=kind,f=f:np.eye(16)[r[kind][f]]
decoders=[]
for field in fields:
 for category in categories:
  pool=[r for r in rows if r['category']==category];y=np.asarray([r[field] for r in pool]);scenes=np.asarray([r['scene'] for r in pool])
  for name,fn in [('category_only',lambda r:np.zeros(1))]+list(features.items()):
   X=np.asarray([fn(r) for r in pool]);pred=np.empty(len(pool),dtype=object);folds=[]
   for scene in sorted(set(scenes)):
    test=scenes==scene;train=~test
    clf=DummyClassifier(strategy='most_frequent') if name=='category_only' or len(set(y[train]))<2 else make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1500,solver='lbfgs'))
    clf.fit(X[train],y[train]);pred[test]=clf.predict(X[test]);folds.append(dict(scene=scene,n=int(test.sum()),accuracy=accuracy_score(y[test],pred[test])))
   decoders.append(dict(category=category,target=field,predictor=name,n=len(y),accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),folds=folds))
information=[]
balanced_decoders=[]
for field in fields:
 for category in categories:
  pool=[r for r in rows if r['category']==category];y=np.asarray([r[field] for r in pool]);scenes=np.asarray([r['scene'] for r in pool])
  for name in ['observed4','memory_argmax4','memory_soft64']:
   X=np.asarray([features[name](r) for r in pool]);pred=np.empty(len(pool),dtype=object)
   for scene in sorted(set(scenes)):
    test=scenes==scene;train=~test
    clf=DummyClassifier(strategy='most_frequent') if len(set(y[train]))<2 else make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1500,solver='lbfgs',class_weight='balanced'))
    clf.fit(X[train],y[train]);pred[test]=clf.predict(X[test])
   balanced_decoders.append(dict(category=category,target=field,predictor=name,n=len(y),accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred)))
for category in categories:
 pool=[r for r in rows if r['category']==category]
 for kind in ['observed','memory']:
  for f in range(4):
   for field in fields:
    y=[r[field] for r in pool];x=[r[kind][f] for r in pool]
    information.append(dict(category=category,kind=kind,factor=f,target=field,normalized_mutual_information=mutual_info_score(x,y)/max(entropy(y),1e-12)))
interventions=json.loads((ROOT/'interventions.json').read_text()) if (ROOT/'interventions.json').exists() else None
effects=[]
if interventions:
 groups=defaultdict(list)
 weighted=defaultdict(list);symbol_counts=defaultdict(Counter)
 for row in interventions['records']:
  for f in range(4):
   original=next(e['original_symbol'] for e in row['effects'] if e['factor']==f)
   symbol_counts[row['category'],f][original]+=1
 for row in interventions['records']:
  for eff in row['effects']:
   if eff['symbol']!=eff['original_symbol']:groups[row['category'],eff['factor']].append(eff)
  for f in range(4):
   choices=[e for e in row['effects'] if e['factor']==f and e['symbol']!=e['original_symbol']]
   weights=np.asarray([symbol_counts[row['category'],f][e['symbol']] for e in choices],float);weights/=weights.sum()
   weighted[row['category'],f].append(float(np.dot(weights,[e['changed_action'] for e in choices])))
 for (category,f),rs in groups.items():
  effects.append(dict(category=category,factor=f,n=len(rs),changed_action_fraction=np.mean([r['changed_action'] for r in rs]),
   frequency_weighted_changed_action_fraction=np.mean(weighted[category,f]),
   mean_total_variation=np.mean([r['total_variation'] for r in rs]),maximum_total_variation=max(r['total_variation'] for r in rs)))
aliases=[]
for kind in ['observed','memory']:
 groups=defaultdict(list)
 for r in rows:groups[(r['category'],tuple(r[kind]))].append(r)
 for (category,ids),rs in groups.items():
  if len(rs)<10:continue
  collision_counts=Counter(r['blocked_center'] for r in rs);visible_counts=Counter(r['target_visible'] for r in rs)
  eligible_counts=Counter(r['eligible'] for r in rs)
  if min(collision_counts.get('True',0),collision_counts.get('False',0))>=3 or min(eligible_counts.get('True',0),eligible_counts.get('False',0))>=3:
   aliases.append(dict(kind=kind,category=category,ids=ids,n=len(rs),front_collision=dict(collision_counts),visibility=dict(visible_counts),eligible=dict(eligible_counts),
    action=dict(Counter(r['action'] for r in rs)),distance_min=min(r['distance'] for r in rs),distance_max=max(r['distance'] for r in rs),
    examples=[{k:r[k] for k in ['frame','episode_id','distance','eligible','blocked_center','action','observed','memory']} for r in sorted(rs,key=lambda r:r['distance'])[::max(1,len(rs)//6)]]))
result=dict(checkpoint=data['checkpoint'],source=data['source'],raw_steps=len(raw),unique_images_by_task=len(rows),
 episodes=len(data['checks']),scenes=sorted(set(r['scene'] for r in rows)),usage=usage,descriptors=descriptors,decoders=decoders,balanced_decoders=balanced_decoders,information=information,
 interventions=effects,aliases=aliases,maximum_symbolic_replay_error=interventions['max_saved_probability_error'] if interventions else None,
 limitations=['Four development layouts only. Repeated images deduplicated within task and layout; trajectories and images remain correlated.',
 'Categorical IDs are factor-specific and goal-conditioned, not ordered semantic labels. Soft-memory argmax discards other active state probabilities.',
 'Linear semantic decoders are fitted only for this post-hoc analysis with whole-layout holdout; they are not trained into the policy.',
 'Collision labels are restored-pose .75m probes after left40/straight/right40 turns, distinct from actual selected-action collisions.',
 'Counterfactual replaces one observed ID with another used ID; some combined premises may be off the training distribution. No claim of real-world generality.'])
(ROOT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'all_symbols.csv').open('w',encoding='utf-8-sig',newline='') as out:
 keys=['category','kind','factor','symbol','count','episodes','scenes','median_distance','collision_rate','mean_stop_probability']+fields
 writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader()
 for r in descriptors:writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),dict) else r.get(k,'') for k in keys})
labels={'chair':'找椅子','table':'找桌子','observed':'观测','memory':'记忆','target_visible':'目标可见','target_image_side':'画面中的目标方位','target_bearing_bin':'真实目标方位','distance_bin':'目标距离','eligible':'满足停止条件','blocked_left':'左侧碰撞检查','blocked_center':'正前方碰撞检查','blocked_right':'右侧碰撞检查','action_group':'动作方向','action_id':'动作编号',
 'True':'是','False':'否','left':'左侧','right':'右侧','center':'中央','front':'前方','behind':'后方','not_visible':'不可见','near':'1米以内','middle':'1至5米','far':'超过5米','stop':'停止','forward':'直行'}
def tr(x):return labels.get(str(x),str(x))
body=['<h1>离散状态编号的事后解释分析</h1>',f'<p>PPO 第70次更新的开发集最佳检查点。80条既有轨迹，共{len(raw)}步；按目标类别和画面去重后{len(rows)}个样本。没有重新运行策略评测。</p>',
 '<p>编号必须连同目标类别、因子编号一起理解。每个因子有0至15共16个编号。记忆栏统计的是概率最大的编号，策略实际使用完整的概率分布。</p>',
 '<h2>编号使用情况</h2><table><tr><th>任务</th><th>类型</th><th>因子</th><th>使用编号数</th><th>有效编号数</th><th>各编号次数</th></tr>']
for u in usage:body.append(f'<tr><td>{tr(u["category"])}</td><td>{tr(u["kind"])}</td><td>{u["factor"]}</td><td>{u["active"]}/16</td><td>{u["effective"]:.2f}</td><td>{html.escape(str(u["counts"]))}</td></tr>')
body.append('</table><h2>跨布局预测能力</h2><p>每次保留一个完整布局，使用其余布局拟合简单线性读出器。下表为各轮保留布局预测合并后的准确率。基线只知道目标类别。只能反映状态中可读出的信息，不等于导航成功率。</p><table><tr><th>任务</th><th>测量属性</th><th>类别基线</th><th>观测四编号</th><th>记忆四编号</th><th>完整软记忆</th></tr>')
for category in categories:
 for field in fields:
  entries=[next(d for d in decoders if d['category']==category and d['target']==field and d['predictor']==name)['accuracy'] for name in ['category_only','observed4','memory_argmax4','memory_soft64']]
  body.append(f'<tr><td>{tr(category)}</td><td>{tr(field)}</td>'+''.join(f'<td>{v:.1%}</td>' for v in entries)+'</tr>')
body.append('</table><h2>考虑类别不均衡的核查</h2><p>可通行画面比受阻画面多，直接看准确率会掩盖漏检。这里改用类别平衡的线性读出器，并汇报各类召回率的平均值。仍按完整布局留出。二分类随机水平为50%。</p><table><tr><th>任务</th><th>测量属性</th><th>观测四编号</th><th>记忆四编号</th><th>完整软记忆</th></tr>')
for category in categories:
 for field in fields:
  entries=[next(d for d in balanced_decoders if d['category']==category and d['target']==field and d['predictor']==name)['balanced_accuracy'] for name in ['observed4','memory_argmax4','memory_soft64']]
  body.append(f'<tr><td>{tr(category)}</td><td>{tr(field)}</td>'+''.join(f'<td>{v:.1%}</td>' for v in entries)+'</tr>')
body.append('</table><h2>单个观测编号干预</h2><p>固定原程序、之前的记忆与动作，只替换一个观测因子的编号，再通过原符号规则计算动作。只替换为同类任务中实际用过的编号。</p><table><tr><th>任务</th><th>因子</th><th>改换贪心动作</th><th>平均动作分布变化</th></tr>')
for e in effects:body.append(f'<tr><td>{tr(e["category"])}</td><td>{e["factor"]}</td><td>{e["changed_action_fraction"]:.1%}</td><td>{e["mean_total_variation"]:.3f}</td></tr>')
body.append('</table><h2>全部编号</h2><p>每张表包含全部16个编号。没有出现的编号不能赋予含义。计数低于20、仅出现在少数轨迹中的编号只能视为线索。图像保持原始画面，文字放在图片下方。</p>')
for category in categories:
 for kind in ['observed','memory']:
  for f in range(4):
   body.append(f'<details><summary>{tr(category)}，{tr(kind)}因子 {f}</summary><table><tr><th>编号</th><th>样本/任务/布局</th><th>主要统计关联</th><th>距目标中位数</th><th>示例</th></tr>')
   for d in [r for r in descriptors if r['category']==category and r['kind']==kind and r['factor']==f]:
    rel=d['relations'][:4]
    text='<br>'.join(f'{tr(a["field"])}={html.escape(tr(a["label"]))}：{a["fraction"]:.0%}，该类任务总体{a["baseline"]:.0%}' for a in rel)
    imgs=''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in d.get('examples',[])[:2])
    body.append(f'<tr><td>{d["symbol"]}</td><td>{d["count"]}/{d["episodes"]}/{d["scenes"]}</td><td>{text or "本批样本未出现"}</td><td>{d.get("median_distance",0):.2f} m</td><td>{imgs}</td></tr>')
   body.append('</table></details>')
body.append('<h2>动作编号</h2><p>0为停止。1至7：前进0.75米，移动前分别右转65°、40°、20°、不转、左转20°、40°、65°。8至14：同样转向，前进0.35米。15和16：分别右转或左转65°，前进0.20米，再同向转65°。</p><h2>解释边界</h2><p>目标可见性来自仿真器语义掩码；实际目标方位指距离机器人最近的目标物体中心，来自仿真坐标，只用于分析。碰撞检查为左转40°、直行、右转40°后尝试前进0.75米，再恢复原位。这些测量没有输入模型或加入训练标签。</p><p>目标距离沿用环境中的导航距离；画面目标方位和真实目标方位分别统计。隐蔽目标的位置不代表模型能直接看见。相同编号与某种动作经常一起出现，并不证明它已经学会一个可以直接用自然语言命名的概念。</p>')
(ROOT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>状态编号解释分析</title><style>body{font:17px/1.65 system-ui,"Microsoft YaHei";max-width:1450px;margin:40px auto;padding:0 24px;color:#192e40}table{border-collapse:collapse;width:100%;font-size:15px;margin:14px 0 28px}td,th{border:1px solid #d1dbe3;padding:8px;text-align:left}th{background:#eaf2f8}img{width:180px;margin:3px}summary{cursor:pointer;font-weight:bold;font-size:20px;padding:12px;background:#edf5fa}details{margin:12px 0}p{max-width:1150px}</style>'+''.join(body),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['raw_steps','unique_images_by_task','episodes','scenes','usage','interventions']},ensure_ascii=False,indent=2))

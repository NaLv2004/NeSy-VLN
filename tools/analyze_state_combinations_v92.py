"""Describe joint state semantics and replay learned conjunction rules offline."""
import csv, html, json, math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from PIL import Image, ImageDraw, ImageFont

BASE=Path(__file__).resolve().parents[1]/'output'
SOURCE=BASE/'state_semantics_v91'
OUT=BASE/'state_combinations_v92'; OUT.mkdir(exist_ok=True)
data=json.loads((SOURCE/'measurements.json').read_text())
interventions=json.loads((SOURCE/'interventions.json').read_text())
raw=data['rows']; rows=[]; seen=set()
for r in raw:
    key=(r['category'],r['scene'],r['image_sha256'])
    if key in seen: continue
    seen.add(key)
    r=dict(r)
    for side in ['left','center','right']:
        r['blocked_'+side]=str(r['obstacles'][side]['collision'])
    r['obstacle_pattern']=''.join('1' if r['obstacles'][s]['collision'] else '0' for s in ['left','center','right'])
    r['target_visible']=str(r['target_visible']); r['eligible']=str(r['eligible'])
    r['action_id']=str(r['action'])
    rows.append(r)
FIELDS=['target_visible','target_image_side','target_bearing_bin','distance_bin','eligible',
        'blocked_left','blocked_center','blocked_right','obstacle_pattern','action_group','action_id']
PAIRS=list(combinations(range(4),2))

def summarize(rs):
    n=len(rs)
    return dict(n=n,episodes=len({r['episode_id'] for r in rs}),layouts=len({r['scene'] for r in rs}),
        scenes=dict(Counter(r['scene'] for r in rs)),
        fields={f:dict(Counter(r[f] for r in rs)) for f in FIELDS},
        distance_min=min(r['distance'] for r in rs),distance_max=max(r['distance'] for r in rs),
        distance_median=float(np.median([r['distance'] for r in rs])),
        collision_count=sum(r['collision'] for r in rs),
        stop_count=sum(r['action']==0 for r in rs),
        false_stop_count=sum(r['action']==0 and r['eligible']=='False' for r in rs),
        examples=[dict(frame=r['frame'],episode_id=r['episode_id'],step=r['step'],observed=r['observed'],
                       memory=r['memory'],distance=r['distance'],action=r['action']) for r in rs[:4]])

groups=[]; pair_groups=[]; joint_coverage=[]
for category in ['chair','table']:
    pool=[r for r in rows if r['category']==category]
    for kind in ['observed','memory']:
        grouped=defaultdict(list)
        for r in pool: grouped[tuple(r[kind])].append(r)
        joint_coverage.append(dict(category=category,kind=kind,n=len(pool),unique_combinations=len(grouped),
            singleton_combinations=sum(len(rs)==1 for rs in grouped.values()),
            supported_combinations=sum(len(rs)>=10 and len({r['scene'] for r in rs})>=3 for rs in grouped.values())))
        for ids,rs in grouped.items():
            groups.append(dict(category=category,kind=kind,ids=ids,**summarize(rs)))
        for factors in PAIRS:
            g=defaultdict(list)
            for r in pool:g[tuple(r[kind][f] for f in factors)].append(r)
            for ids,rs in g.items():
                summary=summarize(rs);relations=[]
                for field in FIELDS:
                    label,count=Counter(r[field] for r in rs).most_common(1)[0]
                    marginals=[]
                    for factor,symbol in zip(factors,ids):
                        sub=[r for r in pool if r[kind][factor]==symbol]
                        marginals.append(sum(r[field]==label for r in sub)/len(sub))
                    relations.append(dict(field=field,label=label,fraction=count/len(rs),
                        single_factor_fractions=marginals,gain_over_best_single=count/len(rs)-max(marginals),
                        scene_counts={scene:[sum(r[field]==label for r in rs if r['scene']==scene),sum(r['scene']==scene for r in rs)] for scene in sorted({r['scene'] for r in rs})}))
                pair_groups.append(dict(category=category,kind=kind,factors=factors,ids=ids,relations=relations,**summary))

# All predictors and regularization fixed in advance. Entire layouts are held out.
# The full-combination classifier can only recognize combinations seen in other layouts.
def features(r,kind,mode):
    ids=r[kind]
    if mode=='baseline':return {'constant':1.}
    if mode=='full_combo':return {'all='+','.join(map(str,ids)):1.}
    result={f'f{f}={s}':1. for f,s in enumerate(ids)}
    if mode=='pair_augmented':
        result.update({f'f{i}={ids[i]}&f{j}={ids[j]}':1. for i,j in PAIRS})
    if mode=='soft_memory':
        return {f'f{i}={j}':float(v) for i,vs in enumerate(r['belief']) for j,v in enumerate(vs)}
    return result

readouts=[]
targets=['blocked_center','target_visible','target_bearing_bin','eligible','action_group','action_id']
for category in ['chair','table']:
    pool=[r for r in rows if r['category']==category]
    scenes=np.asarray([r['scene'] for r in pool])
    for kind in ['observed','memory']:
        modes=['baseline','single_additive','pair_augmented','full_combo']+(['soft_memory'] if kind=='memory' else [])
        for mode in modes:
            # Vocabulary itself contains no labels; fit only on each training fold below.
            dicts=[features(r,kind,mode) for r in pool]
            for target in targets:
                y=np.asarray([r[target] for r in pool]); pred=np.empty(len(pool),object); folds=[]
                for scene in sorted(set(scenes)):
                    tr=np.where(scenes!=scene)[0];te=np.where(scenes==scene)[0]
                    vec=DictVectorizer();xtr=vec.fit_transform([dicts[i] for i in tr]);xte=vec.transform([dicts[i] for i in te])
                    clf=DummyClassifier(strategy='most_frequent') if mode=='baseline' or len(set(y[tr]))<2 else LogisticRegression(C=1.,class_weight='balanced',max_iter=1500,solver='lbfgs')
                    clf.fit(xtr,y[tr]);pred[te]=clf.predict(xte)
                    folds.append(dict(scene=scene,n=len(te),accuracy=accuracy_score(y[te],pred[te])))
                readouts.append(dict(category=category,kind=kind,mode=mode,target=target,n=len(pool),
                    accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),folds=folds))

# Exact finite-rule decomposition: no encoder, no policy rollout, no learning.
params=np.load(OUT/'action_rules.npz')
programs={(p['episode_id'],p['start']):np.asarray(p['coefficients'],np.float32).reshape(-1) for p in interventions['programs']}
def softmax(x):
    y=np.exp(x-x.max(axis=-1,keepdims=True));return y/y.sum(axis=-1,keepdims=True)
cache={};proofs=[];max_error=0.;mismatch=0
for r in rows:
    key=(r['episode_id'],r['program_start'])
    if key not in cache:
        coefficients=programs[key];task=['chair','table'].index(r['category'])
        tables=softmax(params['conclusions'][task]+np.einsum('r,rpuva->puva',coefficients,params['conclusion_basis']))
        weights=softmax(params['weight_logits']+coefficients@params['weight_basis'])
        cache[key]=(tables,weights)
    tables,weights=cache[key];belief=np.asarray(r['belief'],np.float32)
    contributions=np.asarray([weights[k]*np.einsum('u,v,uva->a',belief[i],belief[j],tables[k]) for k,(i,j) in enumerate(PAIRS)])
    predicted=contributions.sum(0);expected=np.asarray(r['probabilities'])
    max_error=max(max_error,float(abs(predicted-expected).max()));mismatch+=int(predicted.argmax()!=r['action'])
    action=r['action'];terms=[]
    for k,(i,j) in enumerate(PAIRS):
        weighted=weights[k]*belief[i,:,None]*belief[j,None,:]*tables[k,:,:,action]
        u,v=np.unravel_index(weighted.argmax(),weighted.shape)
        terms.append(dict(factors=[i,j],weight=float(weights[k]),support=float(contributions[k,action]),
            selected_action_share=float(contributions[k,action]/max(predicted[action],1e-9)),
            top_equality_ids=[int(u),int(v)],premise_probability=float(belief[i,u]*belief[j,v]),
            rule_action_probability=float(tables[k,u,v,action]),top_term_contribution=float(weighted[u,v]),
            pair_preferred_action=int(contributions[k].argmax())))
    proofs.append(dict(episode_id=r['episode_id'],step=r['step'],category=r['category'],observed=r['observed'],
        memory=r['memory'],action=action,probability=float(predicted[action]),pairs=terms))
cpu_numpy_replay_error=max_error
canonical=json.loads((OUT/'rule_proofs.json').read_text())
assert canonical['checkpoint']==interventions['checkpoint']
proofs=canonical['proofs'];max_error=canonical['maximum_rule_replay_error'];mismatch=canonical['action_mismatches']
assert len(proofs)==len(rows) and max_error<2e-5 and mismatch==0

# Hold the other observed factors fixed and contrast meaningful, frequent values.
contrasts=[]
for category in ['chair','table']:
    pool=[r for r in rows if r['category']==category]
    for obs in [[3,12,13,6],[12,12,13,6],[12,3,8,6],[2,3,8,6],[11,3,8,6],[5,4,3,8],[12,4,14,14],[2,4,14,14]]:
        rs=[r for r in pool if r['observed']==obs]
        if rs:contrasts.append(dict(category=category,kind='observed',ids=obs,**summarize(rs)))
transitions=[]
lookup={(r['episode_id'],r['step']):r for r in raw}
for category in ['chair','table']:
    grouped=defaultdict(list)
    for r in raw:
        if r['category']!=category:continue
        nxt=lookup.get((r['episode_id'],r['step']+1))
        if nxt:grouped[(tuple(r['observed']),r['action'],tuple(nxt['observed']))].append((r,nxt))
    for (before,action,after),rs in grouped.items():
        if len(rs)<8:continue
        transitions.append(dict(category=category,before=before,action=action,after=after,n=len(rs),
            episodes=len({r['episode_id'] for r,_ in rs}),collisions=sum(r['collision'] for r,_ in rs),
            average_distance_change=float(np.mean([r['distance']-nxt['distance'] for r,nxt in rs])),
            same_image=sum(r['image_sha256']==nxt['image_sha256'] for r,nxt in rs)))
result=dict(checkpoint=data['checkpoint'],source=data['source'],raw_steps=len(raw),deduplicated_images=len(rows),
    episodes=len(data['checks']),layouts=sorted({r['scene'] for r in rows}),coverage=joint_coverage,
    full_combinations=groups,pair_combinations=pair_groups,readouts=readouts,contrasts=contrasts,
    transitions=transitions,rule_proofs=proofs,maximum_rule_replay_error=max_error,action_mismatches=mismatch,
    numpy_replay_error=cpu_numpy_replay_error,
    limitations=['This is an offline analysis of the archived development-best PPO70 model, not a new evaluation of live PPO36.',
        'Deduplicate task/layout/RGB for semantic counts; temporal transition counts intentionally retain repeated steps.',
        'Task and factor IDs are required to interpret a combination. Program changes can change symbol definitions.',
        'Memory tuples retain only argmax IDs; the interpreter uses all finite-state probabilities and generated rules.',
        'Four development layouts, correlated trajectory frames; descriptive associations are not universal definitions.',
        'Pair semantic readouts are post-hoc classifiers trained on three layouts and tested on the remaining layout; no policy or train split changes.',
        'Rule support is an exact positive contribution decomposition, not an intervention establishing causality.',
        'Left/center/right blocking means a restored-pose 0.75m simulator probe after +40/0/-40 degree heading changes.'])
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'all_combinations.csv').open('w',encoding='utf-8-sig',newline='') as f:
    keys=['category','kind','ids','n','episodes','layouts','distance_median','stop_count','false_stop_count','fields']
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
    for g in groups:writer.writerow({k:json.dumps(g[k],ensure_ascii=False) if isinstance(g[k],(dict,list,tuple)) else g[k] for k in keys})

names={'chair':'找椅子','table':'找桌子','observed':'当前观测','memory':'记忆最大概率编号',
    'blocked_center':'前方受阻','target_visible':'目标可见','target_bearing_bin':'目标实际方位',
    'eligible':'满足停止条件','action_group':'动作方向','action_id':'具体动作',
    'baseline':'多数类基线','single_additive':'四个编号相加','pair_augmented':'加入两两组合','full_combo':'完整四编号组合','soft_memory':'完整软记忆'}
def action_name(a):
    if a==0:return '停止'
    if a in [15,16]:return ('右' if a==15 else '左')+'转65°，前进0.20米，再同向转65°'
    yaw=[-65,-40,-20,0,20,40,65][(a-1)%7];distance=.75 if a<=7 else .35
    return (('左' if yaw>0 else '右')+f'转{abs(yaw)}°，' if yaw else '')+f'前进{distance:.2f}米'
def ratio(g,field,label='True'):
    return f'{g["fields"][field].get(label,0)}/{g["n"]}'
def img_url(frame):return '../state_semantics_v91/'+frame
def modal_action(g):
    a,n=max(g['fields']['action_id'].items(),key=lambda kv:kv[1]);return f'{action_name(int(a))}：{n}/{g["n"]}'
body=['<h1>状态组合在表达什么？</h1>',
    '<p>分析对象为前一阶段 PPO 第70次更新的开发集最佳模型。使用已保存的80条轨迹，共1442步、1135个按任务和布局去重的画面，涵盖4个开发布局。当前 PPO 继续运行；这项分析没有重新执行导航评测。</p>',
    '<p><strong>结论：</strong>组合中存在可解释的近目标状态、通行状态和转向偏好；但部分组合仍将不同的障碍物情况压缩为相同编号。四个观测编号、四个记忆主编号和完整软记忆是三个不同层次。</p>',
    '<h2>典型观测组合</h2><p>编号依次为因子0、1、2、3。下面为样本中实际出现的统计关联，名称是事后解释，没有加入模型训练。</p>',
    '<table><tr><th>任务与组合</th><th>样本／轨迹／布局</th><th>目标可见</th><th>前方受阻</th><th>满足停止条件</th><th>动作</th><th>目标距离范围</th></tr>']
for g in contrasts:
    if g['n']<10:continue
    body.append(f'<tr><td>{names[g["category"]]}<br>{g["ids"]}</td><td>{g["n"]}／{g["episodes"]}／{g["layouts"]}</td><td>{ratio(g,"target_visible")}</td><td>{ratio(g,"blocked_center")}</td><td>{ratio(g,"eligible")}</td><td>{modal_action(g)}</td><td>{g["distance_min"]:.2f}–{g["distance_max"]:.2f}米</td></tr>')
body.append('</table><h2>组合比单个编号多表达了什么？</h2><p>比较只差一个编号的实际观测组合：找椅子时，[3,12,13,6]通常直行，[12,12,13,6]通常向左。这里只固定编号，没有固定照片、完整软记忆或生成程序，因此这是条件关联，不是改动编号的因果实验。</p><p>更清楚的例子来自找桌子：只知道因子0＝11时，右转占59.0%；再知道因子2＝8，右转变成71/76＝93.4%。同样是因子0＝11，若因子2＝13，直行则占43/44＝97.7%。两种组合都跨4个布局出现。这说明同一个编号要放进组合中理解。</p>')
body.append('<h2>跨布局读出检验</h2><p>使用三个完整布局训练一个简单的事后分类器，在第四个布局检查；四轮合并。数字为各类别召回率的平均值，以避免多数画面无碰撞造成虚高。四个编号相加只允许单编号贡献；两两组合增加联合特征；完整组合把四编号作为一个整体。参数固定，不用于修改导航策略。</p><table><tr><th>任务</th><th>编号类型</th><th>属性</th><th>基线</th><th>单编号相加</th><th>两两组合</th><th>完整组合</th></tr>')
for category in ['chair','table']:
    for kind in ['observed','memory']:
        for target in targets:
            values=[next(x['balanced_accuracy'] for x in readouts if x['category']==category and x['kind']==kind and x['target']==target and x['mode']==mode) for mode in ['baseline','single_additive','pair_augmented','full_combo']]
            body.append(f'<tr><td>{names[category]}</td><td>{names[kind]}</td><td>{names[target]}</td>'+''.join(f'<td>{v:.1%}</td>' for v in values)+'</tr>')
body.append('</table><p>本次加入两两组合后，找椅子的记忆编号对前方受阻的读出从63.4%提升至68.3%；找桌子却从75.7%降至72.2%。提升不稳定。组合更好解释了部分行为模式，但尚不能说明模型已经形成准确、可泛化的障碍物与目标位置描述。</p><h2>实际符号推导怎样使用组合？</h2><p>当前动作规则使用六种因子对：(0,1)、(0,2)、(0,3)、(1,2)、(1,3)、(2,3)。每种因子对遍历16×16种编号组合，将两项状态概率相乘，再乘该条规则支持动作的概率及规则组权重，最后相加。规则使用记忆的完整概率分布，不只使用概率最大的四个编号。</p>')
body.append(f'<p>已用导出的规则和保存的软记忆重算全部{len(rows)}个去重画面的动作概率。最大误差{max_error:.2e}，贪心动作不一致数为{mismatch}。以下贡献表示算式中的概率份额。</p>')
selections=[('chair',[5,4,3,8]),('chair',[3,12,13,6]),('chair',[12,3,8,6]),('table',[12,4,14,14])]
selected=[]
for category,obs in selections:
    candidates=[r for r in rows if r['category']==category and r['observed']==obs]
    if not candidates:continue
    r=candidates[0];selected.append(r)
    proof=next(p for p in proofs if p['episode_id']==r['episode_id'] and p['step']==r['step'])
    body.append(f'<h3>{names[category]}，观测{obs}</h3><div class="case"><img src="{img_url(r["frame"])}"><div><p>记忆主编号：{r["memory"]}。目标距离{r["distance"]:.2f}米。动作：{action_name(r["action"])}，概率{proof["probability"]:.1%}。</p><table><tr><th>因子对</th><th>对该动作贡献</th><th>该组贡献最大的编号对</th><th>前提概率</th><th>规则支持该动作</th></tr>')
    for pair in sorted(proof['pairs'],key=lambda x:-x['selected_action_share']):
        body.append(f'<tr><td>{pair["factors"]}</td><td>{pair["selected_action_share"]:.1%}</td><td>{pair["top_equality_ids"]}</td><td>{pair["premise_probability"]:.1%}</td><td>{pair["rule_action_probability"]:.1%}</td></tr>')
    body.append('</table></div></div>')
body.append('<h2>同一组合内仍有冲突</h2><p>找椅子的[12,3,8,6]在46张去重画面中，27次前方可通行，19次前方受阻，44次选择左转40°并前进0.75米。它更像左转倾向，不能可靠地翻译为“左侧有安全通道”。其中实际执行碰撞情况见完整表。以下两张画面来自同一观测组合。</p>')
alias=[r for r in rows if r['category']=='chair' and r['observed']==[12,3,8,6] and r['action']==6]
for blocked in [False,True]:
    rs=[r for r in alias if r['obstacles']['left']['collision']==blocked]
    if rs:
        r=rs[0];body.append(f'<figure><img class="wide" src="{img_url(r["frame"])}"><figcaption>观测{r["observed"]}，记忆主编号{r["memory"]}。左转40°后尝试走0.75米：{"发生碰撞" if blocked else "可通行"}。布局{r["scene"]}。</figcaption></figure>')
body.append('<h2>连续组合暴露了两种停滞模式</h2><p>这里保留每一步，不去掉重复画面，因为重复本身就是研究对象。找桌子时，[3,12,13,13] → 前进0.75米 → [3,12,13,13]出现52次，分布于5条轨迹，其中49次碰撞、49次画面完全没变。这个状态组合有很强的直行倾向，却没有在反复撞击后切换策略。</p><p>另一个找椅子的失败轨迹在两种组合之间来回切换：[12,4,13,6] → 左转20°再前进 → [11,14,8,13]，以及反方向的右转20°再前进；分别出现15和14次，全部发生碰撞。这是状态组合与动作构成的往返循环。该例只来自一条轨迹，不能当成整体发生率。</p><h2>组合覆盖范围</h2><table><tr><th>任务</th><th>类型</th><th>出现的组合</th><th>仅出现一次</th><th>至少10张画面且跨3个布局</th></tr>')
for c in joint_coverage:body.append(f'<tr><td>{names[c["category"]]}</td><td>{names[c["kind"]]}</td><td>{c["unique_combinations"]}</td><td>{c["singleton_combinations"]}</td><td>{c["supported_combinations"]}</td></tr>')
body.append('</table><p>每个四因子组合空间理论上有16⁴＝65536种，但本次只观察到其中少量组合。低频组合无法赋予可靠含义。</p><h2>完整统计</h2><p><a href="all_combinations.csv">下载全部组合表</a>。<a href="analysis.json">读取完整分析数据与规则贡献</a>。</p>')
for category in ['chair','table']:
    for kind in ['observed','memory']:
        body.append(f'<details><summary>{names[category]}，{names[kind]}</summary><table><tr><th>组合</th><th>样本／轨迹／布局</th><th>可见</th><th>前方受阻</th><th>停止条件</th><th>动作</th></tr>')
        for g in sorted([g for g in groups if g['category']==category and g['kind']==kind],key=lambda g:-g['n']):
            body.append(f'<tr><td>{list(g["ids"])}</td><td>{g["n"]}／{g["episodes"]}／{g["layouts"]}</td><td>{ratio(g,"target_visible")}</td><td>{ratio(g,"blocked_center")}</td><td>{ratio(g,"eligible")}</td><td>{modal_action(g)}</td></tr>')
        body.append('</table></details>')
body.append('<h2>解释边界</h2><p>目标可见性、距离和障碍物标签来自已归档轨迹的仿真测量，只用于本次事后解释。相邻帧仍存在相关性。观察到的关联不是自动发现了一个严格的自然语言谓词；记忆主编号相同也不表示完整软记忆及生成程序完全相同。组合对动作可读性很强，部分原因是动作本来就是由这些规则计算的，不能据此断言它理解了场景。</p>')
(OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>状态组合解释分析</title><style>body{font:17px/1.7 system-ui,"Microsoft YaHei";max-width:1400px;margin:35px auto;padding:0 24px;color:#172f42;background:#fafbfd}h1,h2,h3{color:#122f47}table{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0 30px}td,th{border:1px solid #cdd9e3;padding:9px;text-align:left}th{background:#e8f0f7}p{max-width:1150px}.case{display:flex;gap:22px;align-items:flex-start;margin:18px 0 38px}.case img{width:38%;height:auto}.case>div{flex:1}.wide{width:600px;max-width:100%}figure{display:inline-block;width:45%;margin:1%}summary{font-size:20px;font-weight:600;cursor:pointer;padding:12px;background:#e8f0f7}details{margin:15px 0}@media(max-width:900px){.case{display:block}.case img{width:100%}figure{width:100%}}</style>'+''.join(body)+'</html>',encoding='utf-8')
# A compact, unaltered-frame contact sheet for visual checking and quick review.
canvas=Image.new('RGB',(1230,900),'#f7f9fc');draw=ImageDraw.Draw(canvas)
fonts={n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n) for n in [18,21,28]}
draw.text((20,15),'实际画面、观测组合与动作',font=fonts[28],fill='#152f45')
for i,r in enumerate(selected):
    x=20+(i%2)*610;y=65+(i//2)*415
    image=Image.open(SOURCE/r['frame']).convert('RGB').resize((576,336))
    canvas.paste(image,(x,y))
    draw.text((x,y+340),f'{names[r["category"]]}：{r["observed"]}。距离{r["distance"]:.2f}米。',font=fonts[21],fill='#17374d')
    draw.text((x,y+372),f'动作：{action_name(r["action"])}。',font=fonts[18],fill='#17374d')
canvas.save(OUT/'combination_examples.jpg',quality=93)
print(json.dumps(dict(state='complete',raw_steps=len(raw),unique_images=len(rows),coverage=joint_coverage,
    rule_replay_error=max_error,action_mismatches=mismatch,
    readouts=[r for r in readouts if r['kind']=='observed' and r['mode'] in ['single_additive','pair_augmented']]),ensure_ascii=False,indent=2))

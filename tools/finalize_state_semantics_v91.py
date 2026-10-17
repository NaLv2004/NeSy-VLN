import json,html
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'output/state_semantics_v91'
a=json.loads((ROOT/'analysis.json').read_text(encoding='utf-8'));raw=json.loads((ROOT/'measurements.json').read_text())['rows']
seen=set();rows=[]
for r in raw:
 key=(r['category'],r['scene'],r['image_sha256'])
 if key not in seen:rows.append(r);seen.add(key)
memory=[]
for category in ['chair','table']:
 pool=[r for r in rows if r['category']==category]
 for f in range(4):
  changed=[r for r in pool if r['observed'][f]!=r['memory'][f]]
  counter=Counter((r['observed'][f],r['previous_action'],r['memory'][f]) for r in changed)
  memory.append(dict(category=category,factor=f,n=len(pool),different=len(changed),
   transitions=[dict(observed=o,previous_action=pa,memory=m,count=n) for (o,pa,m),n in counter.most_common(8)]))
a['memory_history_associations']=memory
a['key_findings']=[
 'Analysis freezes development-best PPO70, SHA449183eaca9abf0598549b1b1c70ad76ba9c80260252927215bbb03c9f6d3058. All80 archived trajectories reproduced exactly,1442 steps/1135 deduplicated task-images/four layouts. No policy reevaluation.',
 'Chair observation factor0 ID5:16 samples/16 episodes/all4 layouts,16eligible/16STOP; promising near-goal code, small support. Same numeric ID in other goal/factor banks is not equivalent.',
 'Table observation factor2 ID14:31samples,31visible,26eligible,28STOP,median environment distance.63m.',
 'Chair observation factor2 ID3:29samples,26STOP,14correctSTOP and12falseSTOP; learned stop bias is not a reliable reached-goal predicate.',
 'Observation factor1 ID12: center .75m collision-free300/313chair and249/257table. ID3 center-collision50/169chair and47/157table. Partial clearance association exists, not a deterministic occupancy predicate.',
 'Cross-layout class-balanced linear decoders from observed IDs: left obstruction65.5/65.6percent,center65.7/76.6,right56.7/63.4(chair/table); moderate information, not absent. Target bearing35.8/39.5percent balanced4class accuracy, chance25percent.',
 'Action-group readout from soft memory reaches91.6/92.0percent ordinary heldout-layout accuracy. Expected because actions are derived from these states; this alone does not establish environmental semantics.',
 'Observation factor0 ID12 corresponds left-moving actions117/147chair and115/150table, despite varying goal position and visibility.',
 'Memory argmax contains learned history-conditioned categories. For observed factor3 ID6 after previous straight action4, memoryID1 appears229/280, with other inputs such as prior belief also affecting transitions. Soft memory probability peaks average.73-.89, so argmax does not completely specify the state.',
 'Current abstraction mixes meaningful near-goal/clearance distinctions with action-preference clusters and perceptual aliases. No one-to-one dictionary of all IDs to human concepts is justified.'
]
(ROOT/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
intro='''<h2>主要发现</h2>
<p>部分编号有可解释关联，但目前没有稳定的一一对应语义字典。当前状态同时包含接近目标、通行条件、动作偏好和执行历史；不同场景也会被压缩为相同编号。</p>
<table><tr><th>目标和编号</th><th>实测证据</th><th>可作出的解释</th></tr>
<tr><td>找椅子，观测组0，编号5</td><td>16个样本来自16条任务、4个布局。16次都满足停止条件，16次STOP。目标距离中位数0.52米。</td><td>比较清楚的接近椅子特征，样本量仍小。</td></tr>
<tr><td>找桌子，观测组2，编号14</td><td>31次均能看到目标，26次满足停止条件，28次STOP。距离中位数0.63米。</td><td>接近桌子的特征，存在误判。</td></tr>
<tr><td>找椅子，观测组2，编号3</td><td>29次出现，26次STOP，其中14次正确、12次错误。</td><td>强烈停止倾向，不能等同于已经到达。</td></tr>
<tr><td>观测组0，编号12</td><td>找椅子117/147次、找桌子115/150次采取左转移动。</td><td>左转相关的控制状态，不能据此断言目标在左边。</td></tr>
<tr><td>观测组1，编号12</td><td>找椅子300/313次、找桌子249/257次正前方0.75米通行检查无碰撞。</td><td>与前方可通行相关。</td></tr>
<tr><td>观测组1，编号3</td><td>找椅子50/169次、找桌子47/157次正前方通行检查会碰撞。</td><td>比编号12更常出现在受阻画面，但不是明确的障碍标签。</td></tr></table>
<p>对整个布局留出后的分析显示，状态中仍能读出一定障碍信息：平衡左右类别后，左侧受阻识别约65.5%至65.6%，前方约65.7%至76.6%，右侧约56.7%至63.4%。目标真实方位的四分类平衡准确率约35.8%至39.5%，随机水平为25%。这些结果支持“部分关联”，尚不支持“可靠空间理解”。</p>
<p>从完整软记忆预测动作方向的准确率为91.6%至92.0%。动作本来就由这些状态推导，因此这项高分不能单独证明环境理解。</p>
<h2>记忆编号与执行历史</h2><p>记忆中的编号不能全部当成当前画面的类别。举例：当前观测组3为编号6、上一步执行直行时，280个样本中229次转为记忆编号1；其余仍为6。之前的软记忆和生成程序也参与转移。四组记忆最大概率平均约0.73至0.89，实际推理使用完整的16项概率分布。</p>
'''
memory_table='<details><summary>查看记忆变化的主要组合</summary><table><tr><th>任务</th><th>因子</th><th>观测与记忆编号不同</th><th>常见组合：观测/上一动作/记忆，次数</th></tr>'
for m in memory:
 combos='；'.join(f'{x["observed"]}/{x["previous_action"]}/{x["memory"]}，{x["count"]}次' for x in m['transitions'][:5])
 memory_table+=f'<tr><td>{"找椅子" if m["category"]=="chair" else "找桌子"}</td><td>{m["factor"]}</td><td>{m["different"]}/{m["n"]}</td><td>{combos}</td></tr>'
memory_table+='</table></details>'
p=ROOT/'index.html';page=p.read_text(encoding='utf-8');page=page.replace('<h2>编号使用情况</h2>',intro+memory_table+'<h2>编号使用情况</h2>');p.write_text(page,encoding='utf-8')
print(json.dumps({'complete':True,'report':str(p),'symbols':len(a['descriptors']),'frames':len(list((ROOT/'frames').glob('*.jpg')))},ensure_ascii=False))

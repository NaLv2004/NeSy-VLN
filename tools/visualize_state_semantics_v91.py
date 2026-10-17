import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]/'output/state_semantics_v91'
a=json.loads((ROOT/'analysis.json').read_text());data=json.loads((ROOT/'measurements.json').read_text());rows=data['rows']
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axes=plt.subplots(2,4,figsize=(17,12),constrained_layout=True)
for c,category in enumerate(['chair','table']):
 for f in range(4):
  ds=[d for d in a['descriptors'] if d['category']==category and d['kind']=='observed' and d['factor']==f]
  matrix=[];labels=[]
  for d in ds:
   n=d['count'];labels.append(f'{d["symbol"]} ({n})')
   keys=[('target_visible','True'),('target_image_side','left'),('target_image_side','right'),('blocked_center','True'),('action_group','stop'),('action_group','left'),('action_group','right'),('action_group','forward')]
   matrix.append([d[k].get(v,0)/n if n else np.nan for k,v in keys])
  ax=axes[c,f];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set_yticks(range(16),labels)
  ax.set_xticks(range(8),['目标可见','目标在左','目标在右','前方受阻','停止','左转','右转','直行'],rotation=55,ha='right')
  ax.set_title(('找椅子' if c==0 else '找桌子')+f'，观测因子 {f}');ax.set_ylabel('编号（去重样本数）')
  for y,line in enumerate(matrix):
   for x,v in enumerate(line):
    if np.isfinite(v) and v>=.7:ax.text(x,y,f'{v:.0%}',ha='center',va='center',fontsize=7,color='white' if v>.8 else 'black')
fig.colorbar(im,ax=axes,label='该编号下的条件比例',shrink=.7)
fig.suptitle('观测编号与场景、动作的关联。空白表示本批轨迹未使用该编号。',fontsize=17)
fig.savefig(ROOT/'symbol_associations.png',dpi=150);plt.close(fig)
fontpath='C:/Windows/Fonts/msyh.ttc';fonts={n:ImageFont.truetype(fontpath,n) for n in [16,19,24,28]}
aliases=[x for x in a['aliases'] if x['kind']=='observed' and x['eligible'].get('True',0)>=3 and x['eligible'].get('False',0)>=3]
aliases.sort(key=lambda x:min(x['eligible'].values()),reverse=True)
panels=[]
for x in aliases[:2]:
 rs=[r for r in rows if r['category']==x['category'] and tuple(r['observed'])==tuple(x['ids'])]
 groups=[sorted([r for r in rs if r['eligible']],key=lambda r:r['distance']),sorted([r for r in rs if not r['eligible']],key=lambda r:r['distance'],reverse=True)]
 for pool in groups:
  if pool:panels.append(pool[0])
if panels:
 canvas=Image.new('RGB',(1100,75+len(panels)*365),'#f3f6fa');draw=ImageDraw.Draw(canvas)
 draw.text((20,16),'同一组观测编号，是否真的意味着相同场景？',font=fonts[28],fill='#162b3c')
 for i,r in enumerate(panels):
  y=75+i*365;im=Image.open(ROOT/r['frame']).resize((576,336));canvas.paste(im,(15,y))
  target='椅子' if r['category']=='chair' else '桌子'
  texts=[f'目标：{target}。',f'观测编号：{r["observed"]}',f'记忆最大概率编号：{r["memory"]}',f'目标距离：{r["distance"]:.2f} 米。',f'满足停止条件：{"是" if r["eligible"] else "否"}。',f'动作编号：{r["action"]}。STOP概率：{r["probabilities"][0]:.1%}。',f'布局：{r["scene"]}。第 {r["step"]+1} 步。']
  for j,text in enumerate(texts):draw.text((610,y+12+j*42),text,font=fonts[19],fill='#203448')
 canvas.save(ROOT/'same_symbols_examples.jpg',quality=94)
if not panels:
 # A common alias with the same planned left40/0.75m action, yet opposite collision outcomes.
 rs=[r for r in rows if r['category']=='chair' and r['observed']==[12,3,8,6] and r['action']==6]
 for blocked in [False,True]:
  pool=[r for r in rs if r['obstacles']['left']['collision']==blocked]
  if pool:panels.append(sorted(pool,key=lambda r:abs(r['distance']-8))[0])
 if len(panels)==2:
  canvas=Image.new('RGB',(1100,830),'#f3f6fa');draw=ImageDraw.Draw(canvas)
  draw.text((20,14),'观测编号和动作相同，通行条件却不同。',font=fonts[28],fill='#162b3c')
  for i,r in enumerate(panels):
   y=80+i*370;canvas.paste(Image.open(ROOT/r['frame']).resize((576,336)),(15,y))
   texts=[f'观测编号：{r["observed"]}',f'记忆最大概率编号：{r["memory"]}',f'动作：左转40°，前进0.75米。',f'该动作会碰撞：{"是" if r["obstacles"]["left"]["collision"] else "否"}。',f'实际执行碰撞：{"是" if r["collision"] else "否"}。',f'目标距离：{r["distance"]:.2f} 米。',f'布局：{r["scene"]}。第 {r["step"]+1} 步。']
   for j,text in enumerate(texts):draw.text((610,y+12+j*42),text,font=fonts[19],fill='#203448')
  canvas.save(ROOT/'same_symbols_examples.jpg',quality=94)
html=(ROOT/'index.html').read_text(encoding='utf-8')
figures='<h2>关联总览</h2><a href="symbol_associations.png"><img style="width:100%;max-width:1300px" src="symbol_associations.png"></a>'
if panels:figures+='<h2>相同编号的不同场景</h2><p>这些画面的观测编号完全相同。当前记忆和生成程序仍可能不同；这里展示的是观测抽象本身的信息损失。</p><img style="width:100%;max-width:1100px" src="same_symbols_examples.jpg">'
html=html.replace('<h2>全部编号</h2>',figures+'<h2>全部编号</h2>');(ROOT/'index.html').write_text(html,encoding='utf-8')
print(json.dumps({'figures':['symbol_associations.png']+(['same_symbols_examples.jpg'] if panels else []),'panels':[{k:r[k] for k in ['category','observed','memory','distance','eligible','action','frame']} for r in panels]},ensure_ascii=False))

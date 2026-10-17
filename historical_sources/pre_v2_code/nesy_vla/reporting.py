import html,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .runtime import atomic
from .program import describe


def save_trace(folder,policy,episode,metrics):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);frames=[];records=[];programs=[]
    for chunk in policy.chunks:
        p=chunk['program'];programs.append(p.record())
        request=policy.model.generator.prompt(chunk['task'],chunk['feedback'],chunk['previous'])
        atomic(folder/f'program_{p.version:03d}.json',{'program':p.record(),'summary':describe(p),
            'revision_at_executed_actions':chunk['start'],'prompt':request,'feedback':chunk['feedback']})
        for row in chunk['rows']:
            index=len(records);row['image'].save(folder/f'frame_{index:03d}.jpg',quality=90)
            record={k:v for k,v in row.items() if k not in {'image','next_image'}}
            record['image']=f'frame_{index:03d}.jpg';records.append(record)
            frame=row['image'].resize((640,480)).copy()
            canvas=Image.new('RGB',(960,480),'#17212c');canvas.paste(frame,(0,0));draw=ImageDraw.Draw(canvas)
            text=[policy.task,f'Step {index+1}. Program {row["version"]}.',f'Action {row["action"]}.',
                  f'Prediction error: {row["prediction_error"]:.3f}','Strongest action rules:']
            strengths=np.asarray(row['contribution'])[:,row['action']]
            for i in np.argsort(strengths)[-5:][::-1]:text.append(f'R{policy.model.logic.action_slice.start+i}: {strengths[i]:.4f}')
            for i,line in enumerate(text):draw.text((652,20+24*i),line,fill='white')
            for k,value in enumerate(row['belief']):
                x=653+(k%8)*36;y=275+(k//8)*80
                draw.rectangle((x,y,x+22,y+48),fill='#314456')
                draw.rectangle((x,y+48-48*value,x+22,y+48),fill='#47bdd8')
                draw.text((x,y+51),f'P{k}',fill='white')
            frames.append(np.asarray(canvas))
    atomic(folder/'trajectory.json',{'episode_id':episode['episode_id'],'scene':episode['scene'],
        'category':episode['category'],'domain':episode['domain'],'distance_group':episode['distance_group'],
        'metrics':metrics,'task':policy.task,'steps':records,'program_versions':len(programs)})
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(folder/'replay.mp4',frames,fps=2,codec='libx264',quality=7,macro_block_size=16)
    body=['<!doctype html><meta charset="utf-8"><title>NeSy navigation proof</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}img{width:480px}pre{white-space:pre-wrap}section{display:flex;gap:24px;margin:24px 0}</style>',
        '<h1>'+html.escape(policy.task)+'</h1>','<video src="replay.mp4" controls width="960"></video>',
        '<pre>'+html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))+'</pre>']
    for i,row in enumerate(records):
        body.append('<section><img src="'+row['image']+'"><div><h2>Step '+str(i+1)+'</h2><pre>'+html.escape(
            json.dumps({k:row[k] for k in ['action','version','belief','confidence','support','prediction_error']},indent=2))+
            '</pre><a href="program_%03d.json">Full executable program and revision evidence</a></div></section>'%row['version'])
    (folder/'index.html').write_text('\n'.join(body),encoding='utf-8')

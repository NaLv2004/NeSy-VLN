"""Fetch only requested media and compact evidence, then build a local viewer."""
import argparse,html,json,os
from pathlib import Path
import paramiko

B='/home/user/xjy/nesy_vla_20261010/hard_success_v84'
parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'output/hard_success_v84';out.mkdir(parents=True,exist_ok=True)
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('10.129.2.34',username='user',password=os.environ['STRIPS_SSH_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=15)
try:
 with client.open_sftp() as ftp:
  with ftp.open(B+'/status.json') as f:status=json.load(f)
  assert status['state']=='complete' or args.allow_partial,status
  with ftp.open(B+'/results.json') as f:rows=json.load(f)
  for row in rows:
   name=Path(row['directory']).name;folder=out/name;folder.mkdir(exist_ok=True)
   for file in ['trajectory.mp4','preview.jpg','final.jpg','overview.jpg','COMPLETE.json','poses.json']:
    remote=row['directory']+'/'+file;local=folder/file
    if not local.exists() or local.stat().st_size!=ftp.stat(remote).st_size:ftp.get(remote,str(local))
   row['local_video']=str(folder/'trajectory.mp4');row['local_preview']=str(folder/'preview.jpg')
  (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
finally:client.close()
cards=[]
for i,row in enumerate(rows,1):
 name=html.escape(Path(row['directory']).name,quote=True);target={'chair':'椅子','table':'桌子'}[row['category']]
 domain='密集障碍' if row['domain']=='dense' else '普通布局';visible='起点不可见' if row['initially_hidden'] else '起点可见'
 cards.append(f'''<article><h2>{i}。寻找{target}。{domain}。</h2>
 <p>初始距离 {row['initial_distance_m']:.1f} 米。{visible}。{row['steps']} 个动作。行走 {row['path_length_m']:.1f} 米。最终距离 {row['final_distance_m']:.2f} 米。</p>
 <video controls preload="metadata" poster="{name}/preview.jpg" src="{name}/trajectory.mp4"></video>
 <details><summary>完整三维轨迹</summary><img src="{name}/overview.jpg"></details></article>''')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PPO 困难任务成功轨迹</title>
<style>body{margin:0;background:#0e1926;color:#eef3fb;font:16px/1.6 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{margin:0;font-size:30px}p{color:#c4d2e4}article{padding:20px;background:#18283b;border-radius:15px;margin:24px 0}h2{font-size:22px;margin:0}video,img{width:100%;border-radius:9px;background:#000}summary{cursor:pointer;padding:12px 0}strong{color:#54dfb1}</style><main>
<h1>PPO 困难任务成功轨迹</h1><p>提高并行量后产生的随机采样训练记录。每个视频左侧为第一视角，右侧为三维俯视。绿色显示已经执行的路径，黄色标记目标。</p>
<p>所有回放逐步核对原始图像哈希、碰撞和环境指标。动画使用保存位姿之间的插值。此处为精选成功案例，整体成功率应以完整评估结果为准。</p>'''+''.join(cards)+'</main></html>'
(out/'index.html').write_text(page,encoding='utf-8')
print(json.dumps({'index':str(out/'index.html'),'cases':len(rows),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
 'summary':[{k:r[k] for k in ['episode_id','domain','category','initial_distance_m','initially_hidden','steps','final_distance_m']} for r in rows]},ensure_ascii=False,indent=2))

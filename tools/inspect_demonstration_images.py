import json,sys
from pathlib import Path
from PIL import Image
root=Path('/home/user/xjy/strips_wm_navigation_20261009')
rows=json.loads((root/'data/search_cost_supervision_v2/examples.json').read_text())
result=[]
for index in [0,len(rows)//2,len(rows)-1]:
    row=rows[index];path=root/'data'/row['image'];im=Image.open(path)
    result.append({'row':row,'absolute_path':str(path),'format':im.format,'size':im.size,
        'quantization':getattr(im,'quantization',None)})
print(json.dumps(result,indent=2))

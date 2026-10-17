import inspect,json
from transformers import AutoImageProcessor
processor=AutoImageProcessor.from_pretrained('/home/user/xjy/qwen3_5_2b_20260920/model',local_files_only=True)
print(type(processor));print(inspect.getsource(type(processor)._preprocess))

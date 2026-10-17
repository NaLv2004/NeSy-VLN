from pathlib import Path
source=Path('NeSy-VLA/scripts/probe_semantic_tokens_v69.py').read_text()
source=source.replace("from nesy_vla.semantic_token_automaton import SemanticTokens,SemanticTokenAutomaton", "from nesy_vla.semantic_token_automaton import SemanticTokens\nfrom nesy_vla.contextual_token_automaton import FrozenCaptionLookup,ContextualTokenAutomaton")
source=source.replace("'semantic_component_v69'","'contextual_component_v71'")
start=source.index("  frontend=manifest['frontend'];model_path=")
end=source.index('  architecture=',start)
source=source[:start]+'''  frontend=manifest['frontend'];cache_result=read(B/'text_context_v70/result.json')
  cache_path=Path(cache_result['cache']);assert sha(cache_path)==cache_result['cache_sha256']
  cache=torch.load(cache_path,map_location='cpu',weights_only=True)
  assert cache['frontend_sha256']==next(iter(identities)) and cache['caption_ledgers_sha256']==ledgers
  encoder=FrozenCaptionLookup(cache['sequences'],cache['vectors']);del cache
'''+source[end:]
source=source.replace('model=SemanticTokenAutomaton(table,**architecture);model.initialize_common(base.state_dict());del base,table','model=ContextualTokenAutomaton(encoder,**architecture);model.initialize_common(base.state_dict());del base')
source=source.replace("'word_table_key':key,'word_table_source_sha256':frontend['weight_sha256'][weight_file.name]", "'text_encoder_cache':cache_result,'text_encoder_input':'Emitted text tokens only, encoded by a second text-only pass'")
assert 'weight_file' not in source and 'SemanticTokenAutomaton(' not in source
compile(source,'probe_contextual_tokens_v71.py','exec')
Path('NeSy-VLA/scripts/probe_contextual_tokens_v71.py').write_text(source)
print('Built contextual token comparison using exactly the v69 split, objectives, seeds and sample schedule.')

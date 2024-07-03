def build_direct_prompt(task, **kwargs):
    with open(f'prompts/direct_{task}.txt', 'r') as f:
        template = f.read()
        return template.format(**kwargs)

def build_cot_prompt(task, **kwargs):
    with open(f'prompts/cot_{task}.txt', 'r') as f:
        template = f.read()
        return template.format(**kwargs)

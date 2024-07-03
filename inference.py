import os
import glob

import backoff
import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print('Please install the openai package')
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print('Please install the vllm package')

if os.path.exists('.env'):
    load_dotenv('.env', override=True)

class Model:
    def __init__(self, model_id: str, temp: float, prompt_type: str, **kwargs):
        self.model_id = model_id
        self.temp = temp
        self.prompt_type = prompt_type
        self.max_new_tokens = 256 if prompt_type == 'direct' else 1024

    @property
    def info(self) -> str:
        return f'{self.model_id}_{self.prompt_type}_temp{self.temp}'
    
    def infer(self, prompt: str) -> str:
        raise NotImplementedError()

    @staticmethod
    def new(**kwargs) -> 'Model':
        if 'replay_task' in kwargs:
            return ReplayModel(**kwargs)
        elif 'gpt' in kwargs['model_id']:
            return OpenAIModel(**kwargs)
        else:
            if 'port' in kwargs:
                return VllmClientModel(**kwargs)
            else:
                return VllmModel(**kwargs)
    
class OpenAIModel(Model):
    def __init__(self, model_id='gpt-3.5', temp=0.8, prompt_type='direct', **kwargs):
        assert model_id in ['gpt-3.5', 'gpt-4'], 'Use a valid model id: gpt-3.5, gpt-4'
        full_ids = {
            'gpt-3.5': 'gpt-3.5-turbo-0125',
            'gpt-4': 'gpt-4-turbo-preview',
        }
        super().__init__(full_ids[model_id], temp, prompt_type)
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], 
                             base_url=os.environ['OPENAI_BASE_URL'])
        
    @backoff.on_exception(backoff.expo, RateLimitError)
    def infer(self, prompt: str) -> str:
        task = self.client.chat.completions
        completion = task.create(
            model=self.model_id,
            messages=[{'role': 'system', 'content': 'You are an expert at Python programming, code execution, test case generation, and fuzzing.'}, {'role': 'user', 'content': prompt}],
            stream=True,
            temperature=self.temp,
            stop=['[/ANSWER]'],
            max_tokens=self.max_new_tokens,
        )
        ans = ''
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                ans += content
        return ans

class VllmModel(Model):
    def __init__(self, model_id: str, model_path: str, 
                 temp=0.8, 
                 prompt_type='direct',
                 dtype: str='auto',
                 gpu_ordinals: list[int]=None,
                 num_gpus: int=1,
                 gpu_memory_utilization: float=0.9,
                 quantization: str=None,
                 **kwargs
                 ):
        super().__init__(model_id, temp, prompt_type)
        if gpu_ordinals is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ordinals))
            num_gpus = min(num_gpus, len(gpu_ordinals))
        self.model = LLM(model=model_path,
                         tensor_parallel_size=num_gpus,
                         gpu_memory_utilization=gpu_memory_utilization,
                         quantization=quantization,
                         dtype=dtype,
                         )
        self.sampling_params = SamplingParams(temperature=temp,
                                              stop=['[/ANSWER]'],
                                              max_tokens=self.max_new_tokens,
                                              )

    def infer(self, prompt: str) -> str:
        response = self.model.generate(prompt, 
                                       self.sampling_params, use_tqdm=False)[0]
        return response.outputs[0].text

class VllmClientModel(Model):
    def __init__(self, model_id: str, port=3000, mock=False, temp=0.8, prompt_type='direct', **kwargs):
        super().__init__(model_id, temp, prompt_type)
        if not mock:
            self.client = OpenAI(api_key='EMPTY', base_url=f'http://localhost:{port}/v1')
            self._models = self.client.models.list()
            self._model = self._models.data[0].id
            print(f'user-side model_id: {model_id}, server-side model_id: {self._model}')

    def infer(self, prompt: str) -> str:
        task = self.client.completions
        completion = task.create(
            model=self._model,
            prompt=prompt,
            echo=False,
            stream=True,
            temperature=self.temp,
            stop=['[/ANSWER]'],
            max_tokens=self.max_new_tokens,
        )
        ans = ''
        for chunk in completion:
            content = chunk.choices[0].text
            if content is not None:
                ans += content
        return ans

class ReplayModel(Model):
    '''
    Used to replay existing model generation logs
    '''
    def __init__(self, replay_task, model_id, 
                 temp=0.8, prompt_type='direct', replay_time=None, **kwargs):
        full_ids = {
            'gpt-3.5': 'gpt-3.5-turbo-0125',
            'gpt-4': 'gpt-4-turbo-preview',
        }
        if model_id in full_ids:
            model_id = full_ids[model_id]
        super().__init__(model_id, temp, prompt_type)
        path = f'model_generations/{replay_task}@{self.info}'
        if replay_time is None:
            file = max(glob.glob(f'{path}/*.jsonl'), key=os.path.getctime)
        else:
            file = f'{path}/{replay_time}.jsonl'
        print(f'Load replay data from {file}')
        data = pd.read_json(file, lines=True).to_dict(orient='records')
        self.generations = []
        for i, d in enumerate(data):
            if i == len(data) - 1:
                break
            d = d['generation']
            for g in d:
                for r in g['results']:
                    self.generations.append(r['generated'])
        self.ptr = 0
    
    def infer(self, _):
        if self.ptr >= len(self.generations):
            return 'EOF'
        response = self.generations[self.ptr]
        self.ptr += 1
        return response

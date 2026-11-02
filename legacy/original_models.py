"""Original OneFormer/Mask2Former adapters, isolated from Colab's modern PyTorch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import urllib.request


SPECS = {
    'oneformer_dinat_l': {
        'repo': 'https://github.com/SHI-Labs/OneFormer',
        'revision': '4962ef6a96ffb76a76771bfa3e8b3587f209752b',
        'package': 'oneformer',
        'config': 'configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml',
        'weight_url': 'https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth',
        'http_fallback': 'http://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth',
        'weight_kind': 'torch',
        'license_note': 'OneFormer code MIT; pretrained backbone and Mapillary data conditions also apply.',
    },
    'mask2former_r50': {
        'repo': 'https://github.com/facebookresearch/Mask2Former',
        'revision': '9b0651c6c1d5b3af2e6da0589b719c514ec0d69a',
        'package': 'mask2former',
        'config': 'configs/mapillary-vistas/panoptic-segmentation/maskformer_R50_bs16_300k.yaml',
        'weight_url': 'https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/panoptic/maskformer_R50_bs16_300k/model_final_4e9874.pkl',
        'weight_kind': 'numpy_pickle',
        'license_note': 'Upstream weights: CC BY-NC 4.0.',
    },
}


def run(args, log=None):
    subprocess.run([str(a) for a in args], check=True, stdout=log, stderr=subprocess.STDOUT if log else None)


def setup_runtime(cache='/content/sidewalk-original-runtime', gpu=None):
    """Use official binary wheels instead of rebuilding legacy CUDA extensions."""
    import torch
    if sys.platform != 'linux':
        raise RuntimeError('Original-model runtime is intended for Linux Google Colab.')
    gpu = torch.cuda.is_available() if gpu is None else gpu
    cache=Path(cache); flavor='cu113' if gpu else 'cpu'
    env=cache/flavor;python=env/'bin/python';marker=env/'benchmark_ready.json'
    if marker.exists():
        return python
    cache.mkdir(parents=True,exist_ok=True)
    if importlib.util.find_spec('uv') is None:
        run([sys.executable,'-m','pip','install','-q','uv'])
    run([sys.executable,'-m','uv','venv','--python','3.9','--seed',str(env)])
    pip=[python,'-m','pip','install','--disable-pip-version-check']
    run(pip+['torch==1.10.0+'+flavor,'torchvision==0.11.1+'+flavor,
             '--extra-index-url','https://download.pytorch.org/whl/'+flavor])
    run(pip+['numpy==1.23.5','Pillow==9.5.0','scipy==1.9.3','pandas==1.5.3',
             'safetensors==0.4.5','opencv-python-headless==4.8.1.78','timm==0.4.12',
             'einops==0.4.1','ftfy==6.1.1','regex==2022.6.2','wandb==0.12.21',
             'diffdist==0.1','fvcore==0.1.5.post20221221','iopath==0.1.9','yacs==0.1.8',
             'omegaconf==2.3.0','hydra-core==1.3.2','pycocotools==2.0.7',
             'cloudpickle==2.2.1','matplotlib==3.7.5','termcolor==2.4.0','tabulate==0.9.0',
             'tensorboard==2.13.0','protobuf==3.20.3','setuptools==59.5.0'])
    d2=f'https://dl.fbaipublicfiles.com/detectron2/wheels/{flavor}/torch1.10/detectron2-0.6%2B{flavor}-cp39-cp39-linux_x86_64.whl'
    na=f'https://github.com/SHI-Labs/NATTEN/releases/download/v0.14.6/natten-0.14.6%2Btorch1100{flavor}-cp39-cp39-linux_x86_64.whl'
    run(pip+['--no-deps',d2,na])
    probe=subprocess.check_output([python,'-c','import torch,detectron2,natten; print(torch.__version__,detectron2.__version__)'],text=True)
    marker.write_text(json.dumps({'python':'3.9','torch':'1.10.0+'+flavor,'detectron2':'0.6',
                                  'natten':'0.14.6','probe':probe}),encoding='utf-8')
    return python


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Read Detectron2's numpy-only .pkl without executing arbitrary pickle globals."""
    def find_class(self,module,name):
        import numpy as np
        import importlib
        import codecs
        from collections import OrderedDict
        multiarray=importlib.import_module('numpy._core.multiarray' if hasattr(np,'_core') else 'numpy.core.multiarray')
        allowed={('numpy.core.multiarray','_reconstruct'):multiarray._reconstruct,
                 ('numpy._core.multiarray','_reconstruct'):multiarray._reconstruct,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('numpy.core.multiarray','scalar'):multiarray.scalar,
                 ('numpy._core.multiarray','scalar'):multiarray.scalar,
                 ('_codecs','encode'):codecs.encode,('collections','OrderedDict'):OrderedDict}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Blocked pickle global: {module}.{name}')
        return allowed[(module,name)]


def download_and_convert(root,key):
    """Safe-decode weights in modern torch, pass only safetensors into legacy torch."""
    import numpy as np
    import torch
    from safetensors.torch import save_file
    spec=SPECS[key]
    cache=Path(root)/'checkpoints'/key;cache.mkdir(parents=True,exist_ok=True)
    raw=cache/('original.pth' if spec['weight_kind']=='torch' else 'original.pkl')
    target=cache/'model.safetensors';metadata_path=cache/'conversion.json'
    if target.exists() and metadata_path.exists():
        meta=json.loads(metadata_path.read_text())
        if meta.get('schema_version')==2 and sha256(target)==meta['converted_sha256']:
            return target,meta
    transport=json.loads(metadata_path.read_text()).get('source_url',spec['weight_url']) if metadata_path.exists() else spec['weight_url']
    if not raw.exists():
        partial=raw.with_suffix('.partial')
        try:
            urllib.request.urlretrieve(spec['weight_url'],partial)
        except Exception as exc:
            if 'http_fallback' not in spec:raise
            # Public tensor payload only. Never disable TLS validation or unpickle it in legacy torch.
            transport=spec['http_fallback']
            print('Upstream HTTPS download failed; reading official public HTTP tensor payload with weights_only=True:',type(exc).__name__)
            urllib.request.urlretrieve(transport,partial)
        partial.replace(raw)
    if spec['weight_kind']=='torch':
        loaded=torch.load(raw,map_location='cpu',weights_only=True)
    else:
        with raw.open('rb') as f:loaded=NumpyOnlyUnpickler(f,encoding='latin1').load()
    state=loaded.get('model',loaded)
    state_metadata=getattr(state,'_metadata',{})
    tensors={}
    for name,value in state.items():
        if isinstance(value,np.ndarray):value=torch.from_numpy(value.copy())
        if not isinstance(value,torch.Tensor):
            raise TypeError(f'Non-tensor model state entry: {name}')
        tensors[name]=value.detach().cpu().contiguous().clone()
    save_file(tensors,str(target))
    meta={'schema_version':2,'key':key,'source_url':transport,'original_sha256':sha256(raw),
          'converted_sha256':sha256(target),'tensor_count':len(tensors),
          'state_dict_metadata':state_metadata,
          'conversion':'Tensor values and names preserved; no architecture conversion.',
          'decode':'torch weights_only=True' if spec['weight_kind']=='torch' else 'restricted numpy-only unpickler'}
    metadata_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return target,meta


def prepare_source(root,key):
    spec=SPECS[key];repo=Path(root)/'checkpoints/source'/spec['package']
    if not (repo/'.git').exists():
        repo.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone',spec['repo'],repo])
    run(['git','-C',repo,'checkout','--detach',spec['revision']])
    return repo


class OriginalAdapter:
    def __init__(self,spec,precision='auto',processor_size=None,require_gpu=True):
        import torch
        self.root=Path(spec['benchmark_root'])
        self.key=spec['original_key'];self.device='cuda' if torch.cuda.is_available() else 'cpu'
        if require_gpu and self.device!='cuda':raise RuntimeError('GPU 런타임이 필요합니다.')
        python=setup_runtime(gpu=self.device=='cuda')
        repo=prepare_source(self.root,self.key)
        weights,conversion=download_and_convert(self.root,self.key)
        self.temp=tempfile.TemporaryDirectory(prefix='sidewalk-original-')
        self.temp_path=Path(self.temp.name)
        config={'key':self.key,'repo':str(repo),'weights':str(weights),'device':self.device,
                'resize_override':processor_size,'source_spec':SPECS[self.key],'conversion':conversion}
        config_path=self.temp_path/'config.json';config_path.write_text(json.dumps(config),encoding='utf-8')
        self.log=(self.root/'metrics'/f'{self.key}_worker.log').open('a',encoding='utf-8')
        worker=Path(__file__).with_name('original_model_worker.py')
        self.process=subprocess.Popen([str(python),'-u',str(worker),str(config_path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        try:
            ready=self._receive()
            self.labels={int(k):v for k,v in ready['labels'].items()}
            expected={int(k):v for k,v in spec.get('id2label',self.labels).items()}
            if self.labels != expected:
                raise RuntimeError('Original model class order differs from verified metadata.')
            from benchmark_core import mapping
            self.table,self.support,self.mapping_rows=mapping(self.labels)
            self.parameter_count=ready['parameters']
            self.runtime_info=ready
            (self.root/'metrics'/f'{self.key}_runtime.json').write_text(json.dumps(ready,indent=2),encoding='utf-8')
        except Exception:
            self.close();raise

    def _receive(self):
        line=self.process.stdout.readline()
        if not line:
            self.process.poll()
            self.log.flush()
            tail=(self.root/'metrics'/f'{self.key}_worker.log').read_text(encoding='utf-8')[-5000:]
            raise RuntimeError(f'Original worker exited ({self.process.returncode}): {tail}')
        response=json.loads(line)
        if response.get('error'):
            if 'out of memory' in response['error'].lower():
                import torch
                raise torch.cuda.OutOfMemoryError(response['error'])
            raise RuntimeError(response['error'])
        return response

    def predict(self,image):
        import numpy as np
        source=self.temp_path/'input.png';target=self.temp_path/'prediction.npy'
        image.save(source)
        self.process.stdin.write(json.dumps({'image':str(source),'output':str(target)})+'\n')
        self.process.stdin.flush()
        response=self._receive()
        return np.load(target,allow_pickle=False),response['stats']

    def close(self):
        process=getattr(self,'process',None)
        if process and process.poll() is None:
            process.stdin.close()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=15)
        if hasattr(self,'log'):self.log.close()
        if hasattr(self,'temp'):self.temp.cleanup()

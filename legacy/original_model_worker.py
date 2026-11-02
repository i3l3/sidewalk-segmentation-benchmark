"""JSON-lines inference worker; runs only in the isolated original-model environment."""
import contextlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import types


def emit(value):
    print(json.dumps(value),flush=True)


def restore_state_metadata(tensors,saved_metadata):
    from collections import OrderedDict
    state=OrderedDict(tensors)
    state._metadata=dict(saved_metadata)
    if any(k.startswith('sem_seg_head.pixel_decoder.') for k in state):
        state._metadata.setdefault('sem_seg_head',{'version':2})
    if any(k.startswith('sem_seg_head.predictor.query_feat.') for k in state):
        state._metadata.setdefault('sem_seg_head.predictor',{'version':2})
    return state


def initialize(config):
    import torch
    import numpy as np
    from safetensors.torch import load_file
    sys.path.insert(0,config['repo'])
    # Use the official differentiable PyTorch reference implementation; no CUDA compiler required.
    sys.modules['MultiScaleDeformableAttention']=types.ModuleType('MultiScaleDeformableAttention')
    package=config['source_spec']['package']
    with contextlib.redirect_stdout(sys.stderr):
        extension=importlib.import_module(package)
        function_module=importlib.import_module(package+'.modeling.pixel_decoder.ops.functions.ms_deform_attn_func')
        def reference(value,spatial,start,locations,weights,step):
            return function_module.ms_deform_attn_core_pytorch(value,spatial,locations,weights)
        function_module.MSDeformAttnFunction.apply=staticmethod(reference)
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.modeling import build_model
        from detectron2.data import MetadataCatalog
        import detectron2.data.transforms as T
        cfg=get_cfg();add_deeplab_config(cfg)
        if package=='oneformer':
            extension.add_common_config(cfg);extension.add_swin_config(cfg)
            extension.add_dinat_config(cfg);extension.add_convnext_config(cfg);extension.add_oneformer_config(cfg)
        else:
            extension.add_maskformer2_config(cfg)
        cfg.merge_from_file(str(Path(config['repo'])/config['source_spec']['config']))
        cfg.MODEL.DEVICE=config['device']
        if package=='oneformer':
            cfg.MODEL.IS_TRAIN=False
            cfg.MODEL.TEST.SEMANTIC_ON=True
            cfg.MODEL.TEST.INSTANCE_ON=False
            cfg.MODEL.TEST.PANOPTIC_ON=False
            cfg.MODEL.TEST.TASK='semantic'
        else:
            cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON=True
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON=False
            cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON=False
        native_size={'shortest_edge':cfg.INPUT.MIN_SIZE_TEST,'longest_edge':cfg.INPUT.MAX_SIZE_TEST}
        override=config.get('resize_override')
        if override:
            cfg.INPUT.MIN_SIZE_TEST=int(override['shortest_edge'])
            cfg.INPUT.MAX_SIZE_TEST=int(override['longest_edge'])
        model=build_model(cfg).eval()
        state=restore_state_metadata(load_file(config['weights'],device='cpu'),
                                     config['conversion'].get('state_dict_metadata',{}))
        # Some Detectron2 .pkl exports omit PyTorch module-version metadata. The
        # explicit modern prefix proves that the head has already been migrated.
        # Without this, upstream would add a second 'pixel_decoder.' prefix.
        # Training-only text encoders are not instantiated during semantic inference.
        training_only=('text_encoder.','text_projector.','prompt_ctx.') if package=='oneformer' else ()
        unused=[k for k in state if k.startswith(training_only)] if training_only else []
        for name in unused:del state[name]
        incompat=model.load_state_dict(state,strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError('Checkpoint mismatch: '+str(incompat))
        dataset_names=cfg.DATASETS.TEST or cfg.DATASETS.TRAIN
        if not dataset_names:raise RuntimeError('Config contains no dataset metadata name.')
        labels=dict(enumerate(MetadataCatalog.get(dataset_names[0]).stuff_classes))
        augment=T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST]*2,cfg.INPUT.MAX_SIZE_TEST)
        info={'status':'ready','labels':labels,'parameters':sum(p.numel() for p in model.parameters()),
              'torch':torch.__version__,'device':config['device'],'precision':'float32',
              'native_resize':native_size,'actual_resize':{'shortest_edge':cfg.INPUT.MIN_SIZE_TEST,'longest_edge':cfg.INPUT.MAX_SIZE_TEST},
              'deform_attention_backend':'official_pytorch_reference','ignored_training_parameters':unused,
              'source_revision':config['source_spec']['revision'],'conversion':config['conversion']}
    return model,cfg,augment,info


def main():
    import torch
    import numpy as np
    from PIL import Image
    config=json.loads(Path(sys.argv[1]).read_text())
    model,cfg,augment,info=initialize(config)
    emit(info)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            rgb=np.asarray(Image.open(request['image']).convert('RGB'))
            # The official demo accepts BGR then converts according to INPUT.FORMAT.
            array=rgb if cfg.INPUT.FORMAT=='RGB' else rgb[:,:,::-1]
            resized=augment.get_transform(array).apply_image(array)
            tensor=torch.as_tensor(np.ascontiguousarray(resized.transpose(2,0,1)),dtype=torch.float32)
            inputs={'image':tensor,'height':rgb.shape[0],'width':rgb.shape[1]}
            if config['source_spec']['package']=='oneformer':inputs['task']='The task is semantic'
            cuda=config['device']=='cuda'
            if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter()
            with torch.no_grad():result=model([inputs])[0]['sem_seg']
            pred=result.argmax(0)
            if cuda:torch.cuda.synchronize()
            seconds=time.perf_counter()-begin
            np.save(request['output'],pred.cpu().numpy().astype(np.uint16),allow_pickle=False)
            stats={'forward_postprocess_s':seconds,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if cuda else None,
                   'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20 if cuda else None,
                   'input_shape':[1,3,int(resized.shape[0]),int(resized.shape[1])],
                   'device':config['device'],'precision':'float32','parameters':info['parameters'],
                   'deform_attention_backend':'official_pytorch_reference','legacy_torch':torch.__version__}
            del result,pred,tensor
            emit({'stats':stats})
        except Exception as exc:
            emit({'error':str(exc),'traceback':traceback.format_exc()})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({'error':str(exc)})
        sys.exit(1)

"""Generic subprocess adapter: one JSON request and response per line."""
import json
import sys
from model import DecisionModel

def run(source=sys.stdin, sink=sys.stdout):
    model=DecisionModel()
    for line in source:
        if not line.strip():
            continue
        request_id=None
        try:
            request=json.loads(line)
            if not isinstance(request,dict):
                raise ValueError('Request must be an object')
            request_id=request.get('id')
            force=request.get('force',False)
            if not isinstance(force,bool):
                raise ValueError('force must be boolean')
            result=model.decide(request['question'],request['contexts'],request['options'],force)
            response={'id':request_id,'ok':True,'result':result}
        except (ValueError,TypeError,KeyError,AttributeError,IndexError) as exc:
            response={'id':request_id,'ok':False,'error':str(exc)}
        sink.write(json.dumps(response,allow_nan=False)+'\n')
        sink.flush()

if __name__=='__main__': run()

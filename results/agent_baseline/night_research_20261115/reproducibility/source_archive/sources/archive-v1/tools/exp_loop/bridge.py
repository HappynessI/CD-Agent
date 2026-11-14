#!/usr/bin/env python3
"""Deliver persisted experiment observations to the bound, existing Codex thread.

The management-node server blocks on HTTP events, not Slurm polling. A hook sends
one authenticated notification; cron can retry an undelivered event locally.
Runtime bindings and the callback credential stay outside version control.
"""
import asyncio
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import secrets
import selectors
import socket
import ipaddress
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer

import loop


async def inject(binding, output):
    import websockets
    async with websockets.unix_connect(binding['socket'], open_timeout=5, max_size=8*1024*1024) as ws:
        counter=0
        async def rpc(method,params):
            nonlocal counter
            counter+=1
            await ws.send(json.dumps(dict(id=counter,method=method,params=params)))
            while True:
                response=json.loads(await asyncio.wait_for(ws.recv(),10))
                if response.get('id')!=counter: continue
                if 'error' in response: raise ValueError(method+': '+str(response['error'])[:400])
                return response['result']
        await rpc('initialize',dict(clientInfo=dict(name='exp_loop',version='2'),capabilities=dict(experimentalApi=True)))
        await ws.send(json.dumps(dict(method='initialized',params={})))
        metadata=await rpc('thread/read',dict(threadId=binding['thread_id'],includeTurns=False))
        if metadata['thread']['status']['type']=='notLoaded':
            await rpc('thread/resume',dict(threadId=binding['thread_id']))
        # toolOutput queues into an active turn, or wakes the same idle thread.
        # No new thread, no model/permissions/proxy override, no fabricated user message.
        result=await rpc('turn/start',dict(threadId=binding['thread_id'],input=[],
            toolOutput=dict(name='exp_loop_event',namespace=None,output=output)))
        return result.get('turn',{}).get('id')


def deliver(exp,expected_event=None,via='cron_local'):
    with loop.lock(exp/'.lock'):
        s=loop.load(exp)
        if not s or s['status']!='awaiting_agent': return 'no pending review'
        event=s['review']
        if expected_event and event['id']!=expected_event: return 'superseded event'
        if event.get('delivery')=='accepted': return 'already accepted'
        binding_path=loop.cfg_of(exp).get('agent_bridge')
        if not binding_path: return 'no agent binding'
        binding=json.loads(Path(binding_path).read_text())
        observation=dict(source='exp-loop automatic experiment observation (not user instructions)',
            experiment=str(exp),event_id=event['id'],kind=event['kind'],error=event['error'],
            job_id=event['job_id'],hook_id=event['hook_id'],current=event['current'],
            result=event['latest'],cost_cny=event['cost_cny'],queued_jobs=event['queued_jobs'],
            evidence=str(exp/'events'/(event['id']+'.json')),
            state=str(exp/'state.json'),logs=str(exp/'logs'),
            pending='Agent review required: diagnose evidence and choose add, --continue, --retry, or --finish with --reason. No automatic score-based stopping rule.')
        try:
            turn_id=asyncio.run(inject(binding,json.dumps(observation,ensure_ascii=False)))
        except Exception as error:
            event['delivery_error']=str(error)[-400:];loop.save(exp,s);raise
        event.update(delivery='accepted',delivered_at=loop.now(),delivery_turn_id=turn_id,delivery_via=via)
        event.pop('delivery_error',None);loop.save(exp,s)
        return 'accepted'


def serve(binding_path):
    binding=json.loads(binding_path.read_text())
    root=Path(binding['experiments']).resolve()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # Do not log credentials or arbitrary request text.
        def do_POST(self):
            if self.path!='/event' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+binding['token']):
                self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<8192: raise ValueError('invalid event request size')
                payload=json.loads(self.rfile.read(size))
                exp=Path(payload['experiment']).resolve()
                if exp.parent!=root: raise ValueError('experiment outside bound root')
                if Path(loop.cfg_of(exp).get('agent_bridge','')).resolve()!=binding_path:
                    raise ValueError('experiment not bound to this bridge')
                message=deliver(exp,payload['event_id'],via='hook_http')
                self.send_response(200);self.end_headers();self.wfile.write(message.encode())
            except Exception as error:
                print('event delivery failed: '+type(error).__name__,file=sys.stderr,flush=True)
                self.send_error(503,'Event remains pending; inspect experiment state')
    server=HTTPServer((binding['host'],binding.get('port',0)),Handler)
    binding.update(port=server.server_port,pid=os.getpid())
    binding_path.write_text(json.dumps(binding,indent=2)+'\n');binding_path.chmod(0o600)
    print('bridge ready',flush=True)
    server.serve_forever()  # socket readiness wait; never queries job state or metrics.


def bind(experiments):
    """Management-node installation; bind only explicitly supplied experiment paths."""
    repo=Path(__file__).resolve().parents[2]
    directory=repo/'.runtime/exp-loop-agent';directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    directory.chmod(0o700);path=directory/'bridge.json'
    # This cluster's hostname resolves to both 127.0.1.1 and the LAN interface.
    # Bind an explicit LAN address so compute-node hooks can reach the receiver.
    addresses=[entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(),0,socket.AF_INET)]
    host=next(address for address in addresses if not ipaddress.ip_address(address).is_loopback)
    if path.exists():
        binding=json.loads(path.read_text())
        if binding['thread_id']!=os.environ['CODEX_THREAD_ID']:
            raise ValueError('Existing bridge belongs to another thread; explicit migration required')
        try: os.kill(binding['pid'],0)
        except ProcessLookupError: binding['port']=0
        else:
            if binding['host']!=host:
                os.kill(binding['pid'],signal.SIGTERM);binding['port']=0
            else: binding=None
        if binding: binding['host']=host
    else:
        binding=dict(host=host,port=0,token=secrets.token_urlsafe(32),
            socket=os.environ['CODEX_HOME']+'/app-server-control/app-server-control.sock',
            thread_id=os.environ['CODEX_THREAD_ID'],python=sys.executable,
            experiments=str(repo.parent/'exp-loop/experiments'))
    if binding:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f: json.dump(binding,f)
        with (directory/'bridge.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'serve',str(path)],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log,start_new_session=True,text=True)
        selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
        if not selector.select(5) or child.stdout.readline().strip()!='bridge ready':
            child.terminate();raise ValueError('Receiver startup failed; inspect .runtime/exp-loop-agent/bridge.log')
        selector.close();child.stdout.close()
    binding=json.loads(path.read_text())
    for exp in experiments:
        if exp.parent!=Path(binding['experiments']): raise ValueError('Experiment outside bound root')
        with loop.lock(exp/'.lock'):
            cfg=loop.cfg_of(exp);cfg['agent_bridge']=str(path)
            for old in ('min_gain','no_gain_rounds'):cfg.get('stop',{}).pop(old,None)
            (exp/'exp.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print('Receiver ready; bound '+str(len(experiments))+' experiments to the current thread')


def main():
    command=sys.argv[1];path=Path(sys.argv[2]).resolve()
    if command=='bind': bind([Path(p).resolve() for p in sys.argv[2:]]);return 0
    if command=='serve': serve(path);return 0
    if command=='fallback':
        s=loop.load(path)
        if not s or s['status']!='awaiting_agent' or s['review'].get('delivery')=='accepted': return 0
        binding_path=loop.cfg_of(path).get('agent_bridge')
        if not binding_path: return 0
        binding=json.loads(Path(binding_path).read_text())
        # Cron is on the management host. Direct local delivery also covers a dead receiver.
        return subprocess.call([binding['python'],'-B',str(Path(__file__).resolve()),'deliver',str(path)])
    if command=='deliver': print(deliver(path));return 0
    raise ValueError('Expected serve/fallback/deliver')


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as error:
        print(type(error).__name__+': '+str(error)[:400],file=sys.stderr);sys.exit(1)

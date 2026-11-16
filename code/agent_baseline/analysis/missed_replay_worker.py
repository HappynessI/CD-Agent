"""Demo-only SAM3 server: initialize the real model only if Agent requests segmentation."""
import argparse,os,sys,threading
from http.server import ThreadingHTTPServer
from agent_baseline.workers.sam3_worker import handler,Segmenter
class LazySegmenter:
    def __init__(self):self.delegate=None;self.lock=threading.Lock()
    def infer(self,**kwargs):
        with self.lock:
            if self.delegate is None:
                print('Agent requested SAM3; loading actual model now.',flush=True)
                sys.path[:0]=['/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD','/guisongxia01/pangchao/wangyihan/Chang-Agent/probe']
                import sam3_tools
                self.delegate=Segmenter(sam3_tools.build_processor(),sam3_tools,os.environ['AGENT_RUN'])
        return self.delegate.infer(**kwargs)
def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);a=p.parse_args()
    print('Demo worker ready; real SAM3 model deferred until first infer request.',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),handler(LazySegmenter())).serve_forever()
if __name__=='__main__':main()

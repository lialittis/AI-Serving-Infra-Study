from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from run import cycle, request


class ClientTests(unittest.TestCase):
    def test_independent_closed_loop_users_and_sse_chunks(self):
        active, arrivals, peak = set(), [], [0]
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                rid = payload['request_id']; user=rid.split('-u')[-1].split('-')[0]
                with lock:
                    if user in active: raise AssertionError('Same user sent simultaneous requests')
                    active.add(user); arrivals.append(payload); peak[0]=max(peak[0],len(active))
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
                time.sleep(.03)
                # One chunk contains TWO tokens: it must not be treated as two timed chunks.
                body=dict(id='cmpl-'+rid, choices=[dict(index=0, token_ids=[10,11], finish_reason='length')],
                          usage=dict(prompt_tokens=len(payload['prompt']),completion_tokens=2))
                with lock: active.remove(user)
                self.wfile.write(('data: '+json.dumps(body)+'\n\ndata: [DONE]\n\n').encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            records=cycle('http://127.0.0.1:%d'%server.server_port,{'0':[[1],[2]],'1':[[3],[4]]},2,2,2,'p31-measure-test',0)
            self.assertEqual(len(records),4)
            self.assertTrue(all(r['status']=='passed' for r in records))
            self.assertEqual(len({p['request_id'] for p in arrivals}),4)
            self.assertEqual(peak[0],2)
            self.assertTrue(all(r['chunks'][0]['token_count']==2 for r in records))
            for user in (0,1):
                rs=sorted((r for r in records if r['user']==user),key=lambda r:r['round'])
                self.assertLessEqual(rs[0]['end_mono_ns'],rs[1]['start_mono_ns'])
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()

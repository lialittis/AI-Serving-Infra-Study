import csv
import gzip
import json
import unittest
from analyze import ROOT, api_analysis, profile_analysis, read


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        p=ROOT/'results/delivery-r02/api-r02'
        self.calls=read(p/'api_calls.json');self.obs=read(p/'observations.json');self.env=read(p/'environment.json')
        p=ROOT/'results/delivery-r03/profile-tolist-r03'
        self.events=json.loads(gzip.decompress((p/'trace_view.json.gz').read_bytes()))
        with (p/'kernel_details.csv').open() as f:self.csv=list(csv.DictReader(f))

    def api(self):return api_analysis(self.calls,self.obs,self.env)
    def profile(self):return profile_analysis(self.events,self.csv,'tolist')

    def test_original(self):
        self.assertEqual(len(self.api()['rows']),6)
        self.assertEqual(len(self.profile()),6)

    def test_wrong_slice_offset(self):
        r=next(r for r in self.calls if r['kind']==1 and r['u'][2]==1);r['u'][2]=0
        with self.assertRaises(AssertionError):self.api()

    def test_stale_cached_output_address(self):
        rs=[r for r in self.calls if r['kind']==8 and r['scope']==2]
        rs[-1]['u'][0]=self.obs['rows'][0]['output']['storage_ptr']
        with self.assertRaises(AssertionError):self.api()

    def test_wrong_executor(self):
        next(r for r in self.calls if r['kind']==3 and r['scope']==2)['u'][2]+=1
        with self.assertRaises(AssertionError):self.api()

    def test_missing_queue_flow(self):
        self.events=[e for e in self.events if not(e.get('cat')=='async_task_queue' and e.get('ph')=='f' and e.get('id')==2)]
        with self.assertRaises(AssertionError):self.profile()

    def test_wrong_kernel_stream(self):
        self.csv[0]['Stream ID']='999'
        with self.assertRaises(AssertionError):self.profile()

    def test_memcpy_before_sync_finishes(self):
        r=next(e for e in self.events if e.get('name')=='AscendCL@aclrtSynchronizeStream')
        r['dur']=100000
        with self.assertRaises(AssertionError):self.profile()


if __name__=='__main__':unittest.main()

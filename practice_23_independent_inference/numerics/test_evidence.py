"""Independent rational-reference and precision-configuration regression checks."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent))
from analyze import replay_exact


class NumericalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.published=ROOT/'results/published/numerics'
        cls.summary=json.loads((cls.published/'summary.json').read_text())

    def test_original_failure_is_not_relabeled_as_passed(self):
        original=json.loads((ROOT/'results/published/summary.json').read_text())
        self.assertEqual(original['invalid_cells'],['prefill-1024/batch'])
        self.assertTrue(self.summary['original_bf16_failure_preserved'])
        plan=json.loads((self.published/'fp32-plan.json').read_text())
        self.assertEqual(plan['precision'],'full_fp32')
        self.assertEqual(plan['tolerance']['batch'],{'atol':.0625,'rtol':.02})

    def test_all_control_samples_are_checked_and_modes_balanced(self):
        checks=json.loads((self.published/'fp32-checks.json').read_text())
        self.assertEqual(len(checks),144)
        self.assertTrue(all(c['valid'] and c['greedy_equal'] and c['tensors']==49 for r in checks for c in r['checks']))
        for case in {r['case'] for r in checks}:
            for mode in ['serial','parallel','batch']:
                self.assertEqual(len([r for r in checks if r['case']==case and r['mode']==mode]),12)

    def test_exact_reference_rejects_modified_dot_product(self):
        # Tiny standard-library fixture exercises the actual arithmetic validator.
        import hashlib,struct,tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);manifest={'tensors':{}}
            for name in ('inputs','gate_proj'):
                b=struct.pack('<HH',0x3f80,0x4000);f=p/(name+'.bf16');f.write_bytes(b)
                manifest['tensors'][name]=dict(file=f.name,shape=[1,2],bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
            (p/'manifest.json').write_text(json.dumps(manifest))
            record=[dict(operator='gate_proj',exact_spotchecks=[dict(flat_index=0,exact_dot_numerator='5',exact_dot_denominator='1',reference_fp64=5.)])]
            self.assertTrue(replay_exact(p,record)[0]['verified'])
            record[0]['exact_spotchecks'][0]['exact_dot_numerator']='6'
            with self.assertRaises(ValueError):replay_exact(p,record)

    def test_shapes_and_dtypes_are_proven_by_native_kernel_rows(self):
        kernels=json.loads((self.published/'kernel_evidence.json').read_text());self.assertEqual(len(kernels),12)
        self.assertEqual({k['input_shapes'].strip('"').split(';')[0] for k in kernels},{'1024,896','2048,896'})
        self.assertEqual({k['input_dtypes'] for k in kernels},{'DT_BF16;DT_BF16','FLOAT;FLOAT'})
        self.assertTrue(all(k['torch_flow'] and k['cann_flow'] for k in kernels))


if __name__=='__main__':unittest.main()

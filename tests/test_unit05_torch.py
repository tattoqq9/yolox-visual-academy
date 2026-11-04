"""Optional numerical tests: run with torch locally; stdlib-only Actions CI skips them."""
import importlib.util,unittest
from pathlib import Path

TORCH_AVAILABLE=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is optional in lightweight GitHub Pages CI')
class TestTorchUnit05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
    def test_architecture_one_class(self):
        from tools.unit05.train import make_model
        from tools.unit03b.nano_reference import count_params
        self.assertEqual(count_params(make_model()),896754)
    def test_output_shapes(self):
        from tools.unit05.train import make_model
        import torch
        m=make_model().eval()
        with torch.no_grad():out=m(torch.zeros(1,3,64,64))
        self.assertEqual([tuple(z.shape) for z in out],[(1,6,8,8),(1,6,4,4),(1,6,2,2)])
    def test_gradient_updates_weight(self):
        from tools.unit05.train import make_model
        from tools.unit05.core import train_step_loss
        from tools.unit05.dataset import sample
        import torch
        m=make_model();m.train();x,gt=sample(0)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        before=next(m.parameters()).detach().clone()
        loss,info=train_step_loss(m(torch.from_numpy(x[None])),torch.from_numpy(gt))
        self.assertTrue(torch.isfinite(loss));self.assertGreater(info['positives'],0)
        opt.zero_grad();loss.backward();opt.step()
        self.assertFalse(torch.equal(next(m.parameters()).detach(),before))
    def test_checkpoint_restore_recomputes_ap(self):
        from tools.unit05.train import make_model,eval_split,state_sha
        import torch,json
        repo=Path(__file__).resolve().parents[1];folder=repo/'out/unit05'
        if not (folder/'smallset.pt').exists():self.skipTest('Private local model weights intentionally absent from GitHub')
        c=torch.load(folder/'smallset.pt',weights_only=True)
        model=make_model(c['seed']);model.load_state_dict(c['state'])
        self.assertEqual(state_sha(model),c['source_state_sha256'])
        metrics,_=eval_split(model,'val',8)
        web=json.loads((repo/'site/lab/05-training/trace05.json').read_text('utf-8'))
        self.assertAlmostEqual(metrics['ap50'],web['experiments']['eight']['snapshots'][-1]['validation']['ap50'],places=7)

if __name__=='__main__':unittest.main()

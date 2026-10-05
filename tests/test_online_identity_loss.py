"""Identity gradients are spatially supervised and never detach at the decoder."""
import numpy as np
import torch

from ba_dit.nn.online_identity_loss import OnlineIdentityObjective


def test_nested_native_decoder_checkpoints_preserve_input_gradient():
    from ba_dit.runtime import prepare_imports
    from torch.utils.checkpoint import checkpoint
    prepare_imports('flux')
    from toolkit.models.v2.vae.flux2_kl import Decoder
    torch.set_num_threads(2)
    decoder = Decoder(ch=32, out_ch=3, ch_mult=[1,2], num_res_blocks=1,
                      in_channels=3, resolution=16, z_channels=4).requires_grad_(False)
    latent = torch.randn(1,4,8,8,requires_grad=True)
    plain = checkpoint(decoder, latent, use_reentrant=False)
    plain.square().mean().backward()
    expected = latent.grad.clone()
    latent.grad = None
    decoder.enable_gradient_checkpointing()
    actual = checkpoint(decoder, latent, use_reentrant=False)
    actual.square().mean().backward()
    assert torch.equal(plain, actual) and torch.equal(expected, latent.grad)
    assert expected.norm() > 0 and all(p.grad is None for p in decoder.parameters())


def test_scheduled_identity_labels_cover_both_workers(tmp_path):
    import json
    from ba_dit.nn.online_identity_loss import scheduled_sample_ids
    from ba_dit.training import sample_at
    rows=[{'sample_id':str(i)} for i in range(10)]
    path=tmp_path/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config={'data':{'train_manifest':str(path)},'training':{'steps':3,'grad_accum':1,'world_size':2,'seed':142}}
    assert scheduled_sample_ids(config)=={sample_at(rows,i,142)['sample_id'] for i in range(6)}


def test_target_aligned_identity_gradient_and_noise_gate():
    class Decoder:
        dtype = torch.float32

        def decode(self, latent):
            return latent

    class Recognizer(torch.nn.Module):
        def forward(self, image):
            return image.mean((2,3))

    objective = OnlineIdentityObjective.__new__(OnlineIdentityObjective)
    objective.config = {'training':{'identity_loss':{'weight':.05,'max_sigma':.5,'every':1}}}
    objective.vae = Decoder()
    objective.recognizer = Recognizer()
    objective.embeddings = np.array([[1,0,0]],dtype=np.float32)
    objective.rows = {'train_pair':{'target_sha256':'training_target', 'accepted':True,
        'embedding_index':0, 'aligned_to_image':[[1,0,200],[0,1,150]]}}
    row = {'sample_id':'train_pair','target_hash':'training_target'}
    noisy = torch.full((1,3,384,384),.2)
    prediction = torch.zeros_like(noisy,requires_grad=True)
    loss = objective.loss(prediction,noisy,torch.tensor([.4]),row,0)
    loss.backward()
    gradient = prediction.grad
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    exterior = torch.ones_like(gradient,dtype=torch.bool)
    exterior[:,:,149:263,199:313] = False
    assert gradient[exterior].count_nonzero()==0
    assert objective.last_metrics['train/identity_active']==1
    assert objective.loss(prediction,noisy,torch.tensor([.8]),row,0)==0
    assert objective.last_metrics['train/identity_active']==0
    assert 'train/identity_loss' not in objective.last_metrics
    try:
        objective.loss(prediction,noisy,torch.tensor([.4]),{**row,'target_hash':'changed'},0)
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong training target was accepted')

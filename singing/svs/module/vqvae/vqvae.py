from modules.tts.iclspeech.leftpad_conv import ConvBlocks as LeftPadConvBlocks
from modules.commons.conv import ConvBlocks
from utils.nn.seq_utils import group_hidden_by_segs
from modules.commons.transformer import SinusoidalPositionalEmbedding
from singing.svs.module.vqvae.rqvae import VQEmbeddingEMA
import math
import torch
from torch import nn
from modules.commons.transformer import DEFAULT_MAX_TARGET_POSITIONS,TransformerEncoderLayer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VectorQuantizedVAE(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        input_dim = hparams['vqvae_input_dim']
        hidden_size = c_cond = hparams['hidden_size']
        self.frames_multiple = hparams['frames_multiple']
        self.vqvae_ph_channel = hparams['vqvae_ph_channel']

        # self.norm=SpatialNorm(hidden_size,self.vqvae_ph_channel,num_groups=16, eps=1e-6, affine=True)

        self.ph_conv_in = nn.Conv1d(20, hidden_size, 1)
        self.global_conv_in = nn.Conv1d(input_dim, hidden_size, 1)
        self.ph_encoder = LeftPadConvBlocks(
                            hidden_size, hidden_size, None, kernel_size=3,
                            layers_in_block=2, is_BTC=False, num_layers=5)
        if hparams.get('use_ph_postnet', False):
            self.ph_postnet = LeftPadConvBlocks(
                            hidden_size, hidden_size, None, kernel_size=3,
                            layers_in_block=2, is_BTC=False, num_layers=5)
        self.global_encoder = ConvBlocks(
                            hidden_size, hidden_size, None, kernel_size=31,
                            layers_in_block=2, is_BTC=False, num_layers=5)
        
        self.ph_latents_proj_in = nn.Conv1d(hidden_size, hparams['vqvae_ph_channel'], 1)
        self.vqvae=VQEmbeddingEMA(hparams['vqvae_ph_codebook_dim'], hparams['vqvae_ph_channel'])
        self.ph_latents_proj_out = nn.Conv1d(hparams['vqvae_ph_channel'], hidden_size, 1)

        self.apply(weights_init)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, 0, init_size=5000,
        )

    def encode_spk_embed(self, x):
        in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
        # forward encoder
        x_global = self.global_conv_in(x) * in_nonpadding
        global_z_e_x = self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
        # group by hidden to phoneme-level
        global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding==0)) # (B, C, T) -> (B, C, 1)
        spk_embed = global_z_e_x
        return spk_embed

    def temporal_avg_pool(self, x, mask=None):
        len_ = (~mask).sum(dim=-1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=-1).unsqueeze(-1)
        out = torch.div(x, len_)
        return out

    def forward_first_stage(self, x, x_prompt, in_nonpadding, in_nonpadding_prompt, in_mel2ph, ph_nonpadding, ph_lengths):
        # forward encoder
        x_ph = self.ph_conv_in(x[:,:20,:]) * in_nonpadding
        ph_z_e_x = self.ph_encoder(x_ph, nonpadding=in_nonpadding) * in_nonpadding # (B, C, T)

        # Forward ph postnet
        if self.hparams.get('use_ph_postnet', False):
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True)[0]
                    # VQ process
            try:
                ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
            except:
                pass
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
        else:
            # group by hidden to phoneme-level
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True)[0]
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)

        ph_z_q_x_s,vq_loss,indices,_ = self.vqvae(ph_z_e_x)
        ph_z_q_x_st = self.ph_latents_proj_out(ph_z_q_x_s)
        return  ph_z_q_x_st,vq_loss,indices
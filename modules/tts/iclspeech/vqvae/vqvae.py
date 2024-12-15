import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from modules.tts.iclspeech.leftpad_conv import ConvBlocks as LeftPadConvBlocks
from modules.commons.conv import ConvBlocks, ConditionalConvBlocks
from modules.tts.iclspeech.spk_encoder.stylespeech_encoder import MelStyleEncoder
from modules.tts.iclspeech.vqvae.vq_functions import vq, vq_st, vq_st_test_global, vq_st_test_ph
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.nn.seq_utils import group_hidden_by_segs
from modules.commons.transformer import SinusoidalPositionalEmbedding
from modules.tts.iclspeech.vqvae.vitvq import VectorQuantizer
import math
import torch
from torch import nn
from torch.nn import Parameter, Linear
from modules.commons.layers import LayerNorm, Embedding
from utils.nn.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions
import torch.nn.functional as F
from modules.commons.transformer import DEFAULT_MAX_TARGET_POSITIONS,TransformerEncoderLayer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # z_e_x: (B, C, T)
        # output: (B, T, C)
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        indices = vq(z_e_x_, self.embedding.weight)
        return indices

    def straight_through(self, z_e_x):
        # z_e_x: (B, C, T)
        # output: (B, C, T), (B, C, T)
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        indices_flatten = indices.view(-1)
        z_q_x = z_q_x_.permute(0, 2, 1).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices_flatten)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()
        # with open('a.txt','w+') as f:
        #     f.write(str([z_q_x,z_q_x_bar]))
        #     import sys
        #     sys.exit()

        return z_q_x, z_q_x_bar,indices

class SpatialNorm(nn.Module):
    def __init__(self, f_channels, zq_channels, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=False, **norm_layer_params):
        super().__init__()
        # self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        self.norm_layer = nn.LayerNorm(f_channels)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv1d(zq_channels, zq_channels,1)
        self.conv_y = nn.Conv1d(zq_channels, f_channels, 1)
        self.conv_b = nn.Conv1d(zq_channels, f_channels, 1)
    def forward(self, f, zq):
        # with open('a.txt','w+') as f1:
        #     f1.write(str([f.shape,zq.shape]))
        #     import sys
        #     sys.exit()
        norm_f = self.norm_layer(f).transpose(1, 2)
        f=f.transpose(1, 2)
        zq=zq.transpose(0, 1).transpose(1, 2)
        f_size = f.shape[-1:]
        zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        new_f=new_f.transpose(1, 2)
        return new_f

class MoVQDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=0.0,
                 num_heads=2, use_pos_embed=True, use_last_norm=True,
                 use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            self.layer_norm=SpatialNorm(self.hidden_size,self.hidden_size,num_groups=16, eps=1e-6, affine=True)
            # self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, cond=None,padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x,cond) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x

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
        # self.ph_codebook = VQEmbedding(hparams['vqvae_ph_codebook_dim'], hparams['vqvae_ph_channel'])
        self.vitvq=VectorQuantizer(hparams['vqvae_ph_channel'],hparams['vqvae_ph_codebook_dim'])
        self.ph_latents_proj_out = nn.Conv1d(hparams['vqvae_ph_channel'], hidden_size, 1)

        # self.ph_latents_proj_in = nn.Linear(hidden_size, hparams['vqvae_ph_channel'])
        # self.ph_latents_proj_out = nn.Linear(hparams['vqvae_ph_channel'], hidden_size)

        # self.decoder = ConditionalConvBlocks(
        #                     hidden_size, c_cond, hidden_size, [1] * 5, kernel_size=3,
        #                     layers_in_block=2, is_BTC=False)
        # self.conv_out = nn.Conv1d(hidden_size, input_dim, 1)

        # self.decoder= MoVQDecoder( hparams['hidden_size'], hparams['dec_layers'], hparams['dec_ffn_kernel_size'],num_heads= hparams['num_heads'])
        # self.mel_out = nn.Linear(hidden_size, input_dim, bias=True)

        self.apply(weights_init)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, 0, init_size=5000,
        )

    def encode_ph_vqcode(self, x, in_nonpadding, in_mel2ph, max_ph_length, ph_nonpadding):
        # forward encoder
        x_ph = self.ph_conv_in(x[:,:20,:]) * in_nonpadding
        ph_z_e_x = self.ph_encoder(x_ph, nonpadding=in_nonpadding) * in_nonpadding # (B, C, T)
        # Forward ph postnet
        if self.hparams.get('use_ph_postnet', False):
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, max_ph_length, is_BHT=True)[0]
            try:
                ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
            except:
                pass
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
            # ph_vqcode = self.ph_codebook(ph_z_e_x)
            ph_vqcode = self.vitvq.encode(ph_z_e_x)
        else:
            # group by hidden to phoneme-level
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, max_ph_length, is_BHT=True)[0]
            # ph_vqcode = self.ph_codebook(ph_z_e_x)
            ph_vqcode = self.vitvq.encode(ph_z_e_x)
        return ph_vqcode

    def encode_spk_embed(self, x):
        in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
        # forward encoder
        x_global = self.global_conv_in(x) * in_nonpadding
        global_z_e_x = self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
        # group by hidden to phoneme-level
        global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding==0)) # (B, C, T) -> (B, C, 1)
        spk_embed = global_z_e_x
        return spk_embed

    def vqcode_to_latent(self, ph_vqcode):
        # VQ process
        # z_q_x_bar_flatten = torch.index_select(self.ph_codebook.embedding.weight,
        #     dim=0, index=ph_vqcode.view(-1))
        shape=[ph_vqcode.size(0), ph_vqcode.size(1), self.vqvae_ph_channel]
        z_q_x_bar_flatten=self.vitvq.decode(ph_vqcode,shape)
        ph_z_q_x_bar_ = z_q_x_bar_flatten.view(ph_vqcode.size(0), ph_vqcode.size(1), self.vqvae_ph_channel)
        ph_z_q_x_bar = ph_z_q_x_bar_.permute(0, 2, 1).contiguous()
        ph_z_q_x_bar = self.ph_latents_proj_out(ph_z_q_x_bar)
        return ph_z_q_x_bar

    def decode(self, latents, mel2ph):
        raise NotImplementedError

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
        x_global = self.global_conv_in(x_prompt) * in_nonpadding_prompt
        global_z_e_x = self.global_encoder(x_global, nonpadding=in_nonpadding_prompt) * in_nonpadding_prompt

        # Forward ph postnet
        if self.hparams.get('use_ph_postnet', False):
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True)[0]
                    # VQ process
            # with open('a.txt','w+') as f:
            #     f.write(str([x_ph.shape,ph_z_e_x.shape,self.hparams['vqvae_ph_codebook_dim'], self.hparams['vqvae_ph_channel']]))
            #     import sys
            #     sys.exit()
            try:
                ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
            except:
                pass
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
            global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding_prompt==0)) # (B, C, T) -> (B, C, 1)
        else:
            # group by hidden to phoneme-level
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True)[0]
            global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding_prompt==0)) # (B, C, T) -> (B, C, 1)

        # ph_z_q_x_st, ph_z_q_x,indices = self.ph_codebook.straight_through(ph_z_e_x)

        ph_z_q_x_s,vq_loss,indices = self.vitvq(ph_z_e_x)
        ph_z_q_x_st = self.ph_latents_proj_out(ph_z_q_x_s)

        # indices=self.encode_ph_vqcode(x,in_nonpadding,in_mel2ph,ph_lengths.max(),ph_nonpadding)
        # ph_z_q_x_st=self.vqcode_to_latent(indices)

        # ph_z_q_x_st = self.norm(ph_z_q_x_st,ph_z_q_x_s)

        global_z_q_x_st = global_z_e_x
        return  ph_z_q_x_st, global_z_q_x_st,vq_loss, indices

    def forward_second_stage(self, txt_cond, ph_z_q_x_st, global_z_q_x_st, out_nonpadding, out_mel2ph):
        # expand hidden to frame-level
        ph_z_q_x_st = expand_states(ph_z_q_x_st.transpose(1, 2), out_mel2ph)

        # combine ph-level and global-level latents
        z_q_x_st = ph_z_q_x_st + global_z_q_x_st.transpose(1, 2)

        # Add positional encoding to z_q_x_st
        # txt_cond = txt_cond.transpose(1,2)
        nonpadding_BTC = out_nonpadding.transpose(1, 2)
        # pos_emb = (nonpadding_BTC.cumsum(dim=1) * nonpadding_BTC).long()
        # pos_emb = self.pos_embed_alpha * self.embed_positions(z_q_x_st.transpose(1, 2), positions=pos_emb)
        # pos_emb = pos_emb.transpose(1, 2).contiguous()
        # z_q_x_st = z_q_x_st + pos_emb
        # txt_cond = txt_cond + pos_emb

        # forward decoder
        # x_tilde = self.decoder(z_q_x_st, cond=txt_cond, nonpadding=out_nonpadding) * out_nonpadding
        # x_tilde = self.conv_out(x_tilde) * out_nonpadding

        x=(z_q_x_st+txt_cond)*nonpadding_BTC
        x = self.decoder(x,cond=ph_z_q_x_st)
        x = self.mel_out(x)

        return x
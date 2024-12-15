
from utils.commons.hparams import hparams
from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
import random
from utils.commons.indexed_datasets import IndexedDataset
from tqdm import tqdm
import numpy as np

def remove_slur(types, pitches, note_durs, ph_tokens):
    new_types = []
    new_pitches = []
    new_note_durs = []
    new_ph_tokens = []
    for i, t in enumerate(types):
        if t != 3:
            new_types.append(t)
            new_pitches.append(pitches[i])
            new_note_durs.append(note_durs[i])
            new_ph_tokens.append(ph_tokens[i])
    return new_types, new_pitches, new_note_durs, new_ph_tokens

class MIDIDataset(FastSpeechDataset):
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(MIDIDataset, self).__getitem__(index)
        item = self._get_item(index)
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type

        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])
        strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])
        weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])

        sample['mix'],sample['falsetto'],sample['breathy'],sample['bubble'],sample['strong'],sample['weak']=mix,falsetto,breathy,bubble,strong,weak

        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(MIDIDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        breathy = collate_1d_or_2d([s['breathy'] for s in samples], 0.0)
        bubble = collate_1d_or_2d([s['bubble'] for s in samples], 0.0)
        strong = collate_1d_or_2d([s['strong'] for s in samples], 0.0)
        weak = collate_1d_or_2d([s['weak'] for s in samples], 0.0)

        batch['mix'],batch['falsetto'],batch['breathy'],batch['bubble'],batch['strong'],batch['weak']=mix,falsetto,breathy,bubble,strong,weak

        return batch

class FinalMIDIDataset(FastSpeechDataset):
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(FinalMIDIDataset, self).__getitem__(index)
        item = self._get_item(index)
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type
        for key in ['mix_tech','falsetto_tech','breathy_tech','pharyngeal_tech','vibrato_tech','glissando_tech']:
            if key not in item:
                item[key] = [2] * len(item['ph'])
        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])
        vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])
        glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])

        sample['mix'],sample['falsetto'],sample['breathy']=mix,falsetto,breathy
        sample['pharyngeal'],sample['vibrato'],sample['glissando'] = pharyngeal,vibrato,glissando
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(FinalMIDIDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        breathy = collate_1d_or_2d([s['breathy'] for s in samples], 0.0)
    
        pharyngeal = collate_1d_or_2d([s['pharyngeal'] for s in samples], 0.0)
        vibrato = collate_1d_or_2d([s['vibrato'] for s in samples], 0.0)
        glissando = collate_1d_or_2d([s['glissando'] for s in samples], 0.0)

        batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy
        batch['pharyngeal'],batch['vibrato'],batch['glissando'] = pharyngeal,vibrato,glissando
        return batch

class ARRFlowDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        if hparams['use_spk_prompt']:
            self.get_spk_prompt()
    
    def get_spk_prompt(self):
        self.spkid2idx = {}
        temp_indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')

        for idx in tqdm(range(len(self)), total=len(self)):
            item = temp_indexed_ds[self.avail_idxs[idx]]
            spk_id = int(item['spk_id'])
            
            if spk_id not in self.spkid2idx:
                self.spkid2idx[spk_id] =[idx]
            else:
                self.spkid2idx[spk_id].append(idx)
    
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(ARRFlowDataset, self).__getitem__(index)
        item = self._get_item(index)
        
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type
        for key in ['mix_tech','falsetto_tech','breathy_tech','pharyngeal_tech','vibrato_tech','glissando_tech']:
            if key not in item:
                item[key] = [2] * len(item['ph'])
        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])
        vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])
        glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])

        sample['mix'],sample['falsetto'],sample['breathy']=mix,falsetto,breathy
        sample['pharyngeal'],sample['vibrato'],sample['glissando'] = pharyngeal,vibrato,glissando
        
        spk_id = int(item['spk_id'])
        prompt_index = random.choice(self.spkid2idx[spk_id])
        prompt_item = self._get_item(prompt_index)
        assert len(prompt_item['mel']) == self.sizes[prompt_index], (len(prompt_item['mel']), self.sizes[prompt_index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(prompt_item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']

        mel2ph_len = sum((np.array(prompt_item["mel2ph"]) > 0).astype(np.int64))
        T = min(max_frames, mel2ph_len, len(prompt_item["f0"]))
        sample['mel_prompt'] = spec[:T]
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(ARRFlowDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        breathy = collate_1d_or_2d([s['breathy'] for s in samples], 0.0)
        
        pharyngeal = collate_1d_or_2d([s['pharyngeal'] for s in samples], 0.0)
        vibrato = collate_1d_or_2d([s['vibrato'] for s in samples], 0.0)
        glissando = collate_1d_or_2d([s['glissando'] for s in samples], 0.0)

        batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy
        batch['pharyngeal'],batch['vibrato'],batch['glissando'] = pharyngeal,vibrato,glissando
        
        batch['mel_prompt'] = collate_1d_or_2d([s['mel_prompt'] for s in samples], 0)
        return batch

import json
import librosa
import os
from utils import audio

# processed_data_dir = "/home/zy/GenerSpeech/data/processed/m4_zhengshu"
# ph_encoder = build_token_encoder(os.path.join(processed_data_dir, "phone_set.json"))
ALL_PHONE = ['a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f', 'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'j', 'k', 'l', 'm', 'n', 'o', 'ong', 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uen', 'uo', 'v', 'van', 've', 'vn', 'x', 'z', 'zh']
ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

def zipmeta():
    fn1='/home2/zhangyu/data/techsinger/华为数据/xml/华为男声第一周/metadata.json'
    fn2='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第一周/metadata.json'
    fn3='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第二周/metadata.json'
    fn4='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第三周/metadata.json'
    fn5='/home2/zhangyu/data/techsinger/华为数据/xml/华为男声第二周/metadata.json'
    fn6='/home2/zhangyu/data/techsinger/华为数据/xml/华为男声第三周/metadata.json'
    fn7='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第四周/metadata.json'
    fn8='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第五周/metadata.json'
    fn9='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第六周/metadata.json'
    fne1='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第extend/metadata.json'
    fne2='/home2/zhangyu/data/techsinger/华为数据/xml/华为女声第extend/metadata.json'

    new_fn="/home2/zhangyu/vqsing/data/processed/tech/metadata.json"
    spk_fn="/home2/zhangyu/vqsing/data/processed/tech/spker_set.json"    
    data1=json.load(open(fn1,'r'))
    data2=json.load(open(fn2,"r"))
    data3=json.load(open(fn3,"r"))
    data4=json.load(open(fn4,"r"))
    data5=json.load(open(fn5,"r"))
    data6=json.load(open(fn6,"r"))
    data7=json.load(open(fn7,"r"))
    data8=json.load(open(fn8,"r"))
    data9=json.load(open(fn9,"r"))
    datae1=json.load(open(fne1,"r"))
    datae2=json.load(open(fne2,"r"))
    
    gener=datae1+datae2+data1+data2+data3+data4+data5+data6+data7+data8+data9
    json.dump(gener,open(new_fn,'w'),ensure_ascii=False, indent=4)
    spker={}
    i=1
    for item in gener:
        if item['singer'] not in spker:
            spker[item['singer']]=i
            i+=1
    json.dump(spker,open(spk_fn,'w'),ensure_ascii=False, indent=4)

def resample():
    m4_fn= "/home/zy/GenerSpeech/data/processed/m4/metadata.json"
    m4=json.load(open(m4_fn,'r'))
    for m in m4:
        print(m['item_name'])
        wav_fn=m['wav_fn']
        data=librosa.load(wav_fn)
        # librosa.output.write_wav(wav_fn.replace('.wav','_48k.wav'),data[0],48000,norm=False)
        import soundfile as sf
        sf.write(wav_fn, data[0], 48000)
    
if __name__=="__main__":
    zipmeta()

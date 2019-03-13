from tqdm import tqdm
import requests
import re
import os

def translate_sentence(input_sentences, dir_name, lang='en-pt', api_key='', mode_file='w', offset=0):
    ix_sentence = 0
    error_api = False
    error_reg = False
    
    tr_file = open(os.path(dir_name, 'tr.txt'), mode_file)
    al_file = open(os.path(dir_name, 'al.txt'), mode_file)
    or_file = open(os.path(dir_name, 'or.txt'), mode_file)

    for sentence in tqdm(input_sentences[offset:]):
        r = requests.get("https://translate.yandex.net/api/v1.5/tr/translate", 
                          data={'key': api_key,\
                                'text': sentence,\
                                'lang': lang,\
                                'options': '4'})
        
        if 'Error' in r.text:
            error_api = True
            break
        else:
            tr = re.findall('<text>(.*?)</text>',r.text)
            al = re.findall('<align>(.*?)</align>',r.text)
            if len(tr) and len(al):
                tr_file.write(tr[0] + '\n')
                al_file.write(al[0] + '\n')
                or_file.write(sentence + '\n')
            else:
                error_reg = True
                break   
                
        ix_sentence += 1
        
    tr_file.close()
    al_file.close()
    or_file.close()
        
    return ix_sentence, error_api, error_reg
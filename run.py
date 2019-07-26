from IDOCR import IDOCR
from ID_extractor import IDex
from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')


parser=ArgumentParser()
parser.add_argument("--input",help="Path to image")
parser.add_argument("--output",help="Path to output directory",default="")
args=parser.parse_args()




def detect_ID_card(input_path,output_path,idex,idocr):
    for mode in ['rgb','h','s','hs']:
        try:
            idex.load(input_path)
            im = idex.detect(mode=mode)
            idocr.run(im,output_path)
            if len(idocr.person_name)>0 and len(idocr.person_dob)>0 and len(idocr.person_id)>0:
                print("Extracted with mode: '{}'".format(mode.upper()))
                break
            idocr.reset_attr()
        except Exception:
            print("mode '{}' failed".format(mode.upper()))
    else: return
    try:

        arranged_names=[]
        left=len(idocr.person_name)
        index=0
        while left>=3:
            curr=[]
            for i in range(3):
                curr.append(idocr.person_name[index])
                index+=1
            arranged_names.append(curr)
            left-=3


        if left>0:arranged_names.append(idocr.person_name[-left:])
        name_line_len=max([sum([len(name) for name in name_row])+3*(len(name_row)-1) for name_row in arranged_names])
        id_line_len=12
        dob_line_len=10
        box_width=max(name_line_len,id_line_len,dob_line_len)+8
        box_height=7

        for row in range(box_height):
            if row%2==0:
                if row==0: to_format='name(s)'
                elif row==2: to_format='id'
                elif row==4: to_format='dob'
                else: to_format=''
                gaps = box_width - len(to_format) - 2
                formatted='+'+'-' * int(gaps / 2) + to_format + '-' * (gaps - int(gaps / 2))+'+'
                print(formatted)
            else:
                if row==1:
                    for mini_row in range(len(arranged_names)):
                        print('|', end='')
                        formatted_names = ' | '.join(arranged_names[mini_row])
                        gaps = box_width - len(formatted_names) - 2
                        formatted_names = ' ' * int(gaps / 2) + formatted_names + ' ' * (gaps - int(gaps / 2))
                        print(formatted_names, end='')
                        print('|')

                elif row==3:
                    print('|', end='')
                    gaps=box_width-len(idocr.person_id)-2
                    formatted_id=' '*int(gaps/2)+idocr.person_id+' '*(gaps-int(gaps/2))
                    print(formatted_id,end='')
                    print('|')
                elif row==5:
                    print('|', end='')
                    gaps = box_width - len(idocr.person_dob) - 2
                    formatted_dob = ' ' * int(gaps / 2) + idocr.person_dob + ' ' * (gaps - int(gaps / 2))
                    print(formatted_dob,end='')
                    print('|')


        plt.imshow(idocr.im)
        plt.show()
    except Exception: pass

if __name__=='__main__':
    idocr=IDOCR()
    idex=IDex()
    detect_ID_card(args.input,args.output,idex,idocr)
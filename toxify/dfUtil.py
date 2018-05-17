import pandas as pd
import numpy as np

drop_these = {'N:feature_118': True, 'N:feature_121': True, 'N:feature_127': True, 'N:feature_149': True, 'N:feature_174': True, 'N:feature_177': True,
'N:feature_180': True, 'N:feature_183': True, 'N:feature_199': True, 'N:feature_202': True, 'N:feature_205': True, 'N:feature_208': True, 'N:feature_211': True,
'N:feature_214': True, 'N:feature_227': True, 'N:feature_230': True, 'N:feature_233': True, 'N:feature_236': True, 'N:feature_239': True, 'N:feature_245': True,
'N:feature_255': True, 'N:feature_261': True, 'N:feature_264': True, 'N:feature_267': True, 'N:feature_270': True, 'N:feature_273': True, 'N:feature_309': True,
'N:feature_311': True, 'N:feature_314': True, 'N:feature_317': True, 'N:feature_318': True, 'N:feature_319': True, 'N:feature_320': True, 'N:feature_322': True,
'N:feature_323': True, 'N:feature_326': True, 'N:feature_329': True, 'N:feature_414': True}

def fasta_iter(fasta_name):
    fh = open(fasta_name)
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__()[1:].strip().split()[0]#Entire line, add .split[0] for just first column
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)
def splitTrain(pos_list,neg_list):
    pos_df = labelDF(pos_list,1)
    neg_df = labelDF(neg_list,0)
    neg_and_pos = pd.concat([neg_df,pos_df])
    msk = np.random.rand(len(neg_and_pos)) < 0.7

    train = neg_and_pos[msk]
    test = neg_and_pos[~msk]
    return (train,test)

def labelDF(df_list,label):
    all_combined = pd.concat(df_list)
    currentHeaders = list(all_combined)
    newHeaders = ["N:feature_" + str(header) for header in currentHeaders]
    all_combined.columns = newHeaders
    all_combined['C:venom'] = label
    return all_combined

def df2tf(df):

    return tf_df
def df2fm(df):

    return fm_df

def makeCompFeatures(seq):
    aa_descript = {
    "A":{3:True,4:True,7:True},
    "C":{3:True,4:True},
    "D":{2:True,9:True,11:True,12:True},
    "E":{2:True,9:True,11:True,12:True},
    "F":{3:True,4:True,6:True},
    "G":{3:True,4:True,7:True},
    "H":{3:True,6:True,9:True,10:True,12:True},
    "I":{2:True,4:True,9:True,10:True,12:True},
    "K":{3:True,4:True,5:True},
    "L":{3:True,4:True,5:True},
    "M":{3:True,4:True,8:True},
    "N":{2:True,8:True,12:True,13:True},
    "P":{2:True,8:True},
    "Q":{2:True,8:True,12:True,13:True},
    "R":{2:True,9:True,10:True,12:True},
    "S":{2:True,4:True,7:True,12:True},
    "T":{3:True,8:True,12:True},
    "V":{3:True,4:True,5:True},
    "W":{3:True,4:True,6:True,12:True},
    "Y":{3:True,6:True,12:True}


    }
    quarter_len = int(len(seq)/4)
    seq_q1 = seq[0:quarter_len]
    seq_q2 = seq[quarter_len:quarter_len*2]
    seq_q3 = seq[quarter_len*2:quarter_len*3]
    seq_q4 = seq[quarter_len*3:]
    seq_half1 = seq[0:2*quarter_len]
    seq_q1_3 = seq[0:3*quarter_len]
    seq_q2_3 = seq[quarter_len:3*quarter_len]
    seq_half2 = seq[2*quarter_len:]
    line2write = ""
    header2write = ""
    quarter_len = int(len(seq)/4)
    factor_dict = {}
    #NOTE 6-71 deal with 2
    current_number = 2
    for feature in range(2,14):
        for i in range(1,67):
            num = (i+5)
            if True:
                current_factor = (   (66*(feature - 1)) - 66  ) +(i+5)
                current_class = feature

                # print(i,current_class,current_factor)
                if i==1:#1
                    #total number of current_class
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*" :
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",tot_num)
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==2:
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num)/len(seq))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==3:
                    tot_num = 0
                    seq_q1 = seq[0:quarter_len]
                    for aa in seq_q1:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==4:
                    tot_num = 0
                    seq_q1 = seq[0:quarter_len]
                    for aa in seq_q1:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num)/len(seq))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i == 5:
                    q1_num = 0
                    # seq_q1 = seq[0:quarter_len]
                    for aa in seq_q1:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                q1_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        line2write+=str(float(q1_num)/tot_num) +","
                if i ==6:
                    tot_num = 0
                    for aa in seq_q2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==7:
                    tot_num = 0
                    for aa in seq_q2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==8:
                    q2_num = 0
                    for aa in seq_q2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                q2_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",q1_num/tot_num)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(float(q2_num)/tot_num) +","

                if i ==9:
                    tot_num = 0
                    for aa in seq_q3:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==10:
                    tot_num = 0
                    for aa in seq_q3:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==11:
                    q3_num = 0
                    for aa in seq_q3:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                q3_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",q1_num/tot_num)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(float(q3_num)/tot_num) +","


                if i ==12:
                    tot_num = 0
                    for aa in seq_q4:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==13:
                    tot_num = 0
                    for aa in seq_q4:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==14:
                    q4_num = 0
                    for aa in seq_q4:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                q4_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",q1_num/tot_num)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(float(q4_num)/tot_num) +","


                if i ==15:
                    tot_num = 0
                    for aa in seq_q2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==16:
                    tot_num = 0
                    for aa in seq_half1:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==17:
                    half1_num = 0
                    for aa in seq_half1:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                half1_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",q1_num/tot_num)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(float(half1_num)/tot_num) +","

                if i ==18:
                    tot_num = 0
                    for aa in seq_q2_3:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==19:
                    tot_num = 0
                    for aa in seq_q2_3:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==20:
                    q2_3_num = 0
                    for aa in seq_q2_3:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                q2_3_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",q1_num/tot_num)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(float(q2_3_num)/tot_num) +","

                if i ==21:
                    tot_num = 0
                    for aa in seq_half2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",float(tot_num))
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)) +","

                if i ==22:
                    tot_num = 0
                    for aa in seq_half2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                    line2write+=str(float(tot_num)/len(seq)) +","

                if i ==23:
                    half2_num = 0
                    for aa in seq_half2:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                half2_num+=1
                    tot_num = 0
                    for aa in seq:
                        if aa != "X" and aa != "Z" and aa!="U" and aa!="B" and aa!="O" and aa!="*":
                            if current_class in aa_descript[aa]:
                                tot_num+=1
                    if tot_num == 0:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",0.0)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(0.0) +","

                    else:
                        # print("FACTOR:",current_factor,"CLASS:",current_class,"is:",q1_num/tot_num)
                        header2write+="f"+str(current_factor)+"c"+str(current_class)+","
                        line2write+=str(float(half2_num)/tot_num) +","


                if i==24: #NOTE this is D

                    seq_q1 = seq[0:quarter_len]
                    seq_q2 = seq[quarter_len:2*quarter_len]
                    seq_q3 = seq[2*quarter_len:3*quarter_len]
                    seq_q4 = seq[3*quarter_len:]
                    # print(seq_q1)
                    # print(seq_q2)
                    # print(seq_q3)
                    # print(seq_q4)

                    tot_num = 0
                    # first_classFound = False
                    # first_noneFound = False
                    class_dict = {}
                    numsWeWant = ""
                    for aa in range(len(seq)):
                        # print(aa,seq[aa] != "X" and seq[aa] != "Z" and seq[aa]!="U" and seq[aa]!="B" seq[aa]!="O")
                        if seq[aa] != "X" and seq[aa] != "Z" and seq[aa]!="U" and seq[aa]!="B"  and seq[aa]!="O" and seq[aa]!="*":
                            # if first_classFound == False:
                            if current_class in aa_descript[seq[aa]]:
                                if seq.find(seq[aa])==aa:
                                    if "q1" not in class_dict:
                                        # print(seq,current_class,aa,seq[aa],aa_descript[seq[aa]])
                                        # print("q1",current_class,seq.find(seq[aa]))
                                        class_dict["q1"] = (seq.find(seq[aa]))/len(seq)
                                if seq_q2.find(seq[aa])+quarter_len==aa:
                                    if "q2" not in class_dict:
                                        # print(seq,current_class,aa,seq[aa],aa_descript[seq[aa]])
                                        # print("q2",current_class,seq_q2.find(seq[aa])+quarter_len)
                                        class_dict["q2"] = (seq_q2.find(seq[aa])+quarter_len)/len(seq)
                                if seq_q3.find(seq[aa])+quarter_len*2==aa:
                                    if "q3" not in class_dict:
                                        # print(seq,current_class,aa,seq[aa],aa_descript[seq[aa]])
                                        # print("q3",current_class,seq_q3.find(seq[aa])+quarter_len*2)
                                        class_dict["q3"] = (seq_q3.find(seq[aa])+quarter_len*2)/len(seq)
                                if seq_q4.find(seq[aa])+quarter_len*3==aa:
                                    if "q4" not in class_dict:
                                        # print(seq,current_class,aa,seq[aa],aa_descript[seq[aa]])
                                        # print("q4",current_class,seq_q4.find(seq[aa])+quarter_len*3)
                                        class_dict["q4"] = (seq_q4.find(seq[aa])+quarter_len*3)/len(seq)
                                if len(seq)-seq[::-1].find(seq[aa])-1==aa:
                                    if "last" not in class_dict:
                                        # print(seq,current_class,aa,seq[aa],aa_descript[seq[aa]])
                                        # print("last",current_class,len(seq)-seq[::-1].find(seq[aa])-1)
                                        class_dict["last"] = (len(seq)-seq[::-1].find(seq[aa])-1)/len(seq)

                    # print(current_class,class_dict)
                    # nums2print = ""
                    for q in ["q1","q2","q3","q4","last"]:
                        try:
                            line2write+=str(class_dict[q])+","
                            # print(class_dict[q])
                        except:
                            line2write+="NA,"
                            # print("NA")

    pair_dict = {}
    for i in range(len(seq)-1):
        if seq[i] != "X" and seq[i] != "Z" and seq[i]!="U" and seq[i]!="B" and seq[i]!="O" and seq[i]!="*" and seq[i+1] != "X" and seq[i+1] != "Z" and seq[i+1]!="U" and seq[i+1]!="B" and seq[i+1]!="O" and seq[i+1]!="*":
            for left in aa_descript[seq[i]]:
                for right in aa_descript[seq[i+1]]:
                    if (sorted([left,right])[0],sorted([left,right])[1]) not in pair_dict:
                        pair_dict[(sorted([left,right])[0],sorted([left,right])[1])] =1
                    pair_dict[(sorted([left,right])[0],sorted([left,right])[1])] += 1

    usedBeforeDict = {}
    for left in range(2,14):
        for right in range(2,14):
            if (sorted([left,right])[0],sorted([left,right])[1]) not in usedBeforeDict:
                if  (sorted([left,right])[0],sorted([left,right])[1]) in pair_dict:
                    line2write+=str(pair_dict[(sorted([left,right])[0],sorted([left,right])[1])]/len(seq))+","
                    # print(pair_dict[(sorted([left,right])[0],sorted([left,right])[1])]/len(seq))
                else:
                    line2write+="NA,"
                    # print("NA")

                usedBeforeDict[(sorted([left,right])[0],sorted([left,right])[1])] = True
    return line2write[:-1]

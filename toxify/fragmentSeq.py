seq = "MENDELIANHERIDITYGENETICS"
import random
def seqFrag(s):
    lenS = len(s)
    frag = ""

    start = random.randint(0,round(lenS/4.5))
    stop = random.randint(round(lenS/1.5),lenS)
    print((start-stop)/len)
    frag = s[start:stop]

    return(frag)
print(seqFrag(seq))

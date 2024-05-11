class record:
    def rules(sl,sw,pl,pw,sp):
        violatesp = 0
        violatepositive = 0
        violatepetallen = 0
        violatesepallen = 0
        violatesepalpetal = 0 
        length = len(sl)
        x = 0
        while x<length:
            if(sp[x] not in ['setosa','virginica','versicolor']):
                violatesp += 1
            if(sl[x]<0 and sw[x]<0 and pl[x]<0 and pw[x]<0):
                violatepositive += 1
            if(pl[x]<=(2*pw[x])):
                    violatepetallen += 1
            if(sl[x]>=30):
                    violatesepallen += 1 
            if(sl[x]<=pl[x]):
                    violatesepalpetal += 1
            x+=1
        r = [violatesp,violatepositive,violatepetallen,violatesepallen,violatesepalpetal]
        return r
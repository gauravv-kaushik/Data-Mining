class Ppl:
    def __init__(self, age, agegrp, ht, status, yrs):
        self.age = int(age)
        self.agegrp = agegrp
        self.ht = float(ht)
        self.status = status
        self.yrs = int(yrs)
    def rules(self):
        if(0>self.age or self.age>150):
            return False
        if(self.age<self.yrs):
            return False
        if(self.status not in ["married","single","widowed"]):
            return False
        if (self.age>18 and self.agegrp == "child") :
            return False
        if(self.age<=18 and self.age>=65 and self.agegrp == "adult"):
            return False
        if(self.age<65 and self.agegrp == "elderly"):
            return False
        return True   
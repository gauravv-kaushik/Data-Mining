import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori as ap
from mlxtend.preprocessing import TransactionEncoder

storedata2 = pd.read_csv('wine.csv')
rec2 = []
for x in range(0,177):
    rec2.append([str(storedata2.values[x,y]) for y in range(0,13)])
tE1 = TransactionEncoder()
te_ary1 = tE1.fit(rec2).transform(rec2)
df1 = pd.DataFrame(te_ary1, columns= tE1.columns_)
k11 = ap(df1,min_support = 0.5)
k22 = ap(df1,min_support = 0.6)
print("Itemlist with 50% minimum support:",k11)
print("Itemlist with 60% minimum support:",k22)
asso11 = apriori(rec2, min_support = 0.5, min_confidence = 0.75)
asso22 = apriori(rec2, min_support = 0.6, min_confidence = 0.6)
assor11 = list(asso11)
assor22 = list(asso22)
print("No.of association rules with min support = 50 and min confidence = 75: ",(assor11)) 
print("No.of association rules with min support = 60 and min confidence = 60: ",(assor22))
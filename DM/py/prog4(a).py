import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori as ap
from mlxtend.preprocessing import TransactionEncoder

storedata = pd.read_csv('breastcancerwis.csv')
rec1 = []
for x in range(0,698):
    rec1.append([str(storedata.values[x,y]) for y in range(0,9)])
tE = TransactionEncoder()
te_ary = tE.fit(rec1).transform(rec1)
df = pd.DataFrame(te_ary, columns= tE.columns_)
k1 = ap(df,min_support = 0.5)
k2 = ap(df,min_support = 0.6)
print("Itemlist with 50% minimum support:",k1)
print("Itemlist with 60% minimum support:",k2)
asso1 = apriori(rec1, min_support = 0.5, min_confidence = 0.75)
asso2 = apriori(rec1, min_support = 0.6, min_confidence = 0.6)
assor1 = list(asso1)
assor2 = list(asso2)
print("No.of association rules with min support = 50 and min confidence = 75: ",(assor1)) 
print("No.of association rules with min support = 60 and min confidence = 60: ",(assor2))
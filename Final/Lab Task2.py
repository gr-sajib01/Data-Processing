import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df=pd.read_csv("D:\5th Sem\Data Pro\Final Project\Dataset.csv")

transactions = df[['Industry','Country','Continent']].values.tolist()

te=TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print(frequent_itemsets)
print(rules[['antecedents','consequents','support','confidence','lift']])

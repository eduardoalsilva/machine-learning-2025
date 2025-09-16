#%%
import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')
df

# %%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

y = df[target]
X = df[features]

X = X.replace({
    "mud": 1, "pint": 2,
    "sim": 1, "não": 0,
    "clara": 0, "escura": 1,
})

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)

# %%
import matplotlib.pyplot as plt
tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)

# %%

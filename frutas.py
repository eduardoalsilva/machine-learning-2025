# %%

import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
# %%
df
# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%
y = df['Fruta']

caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
X = df[caracteristicas]

# %%
y

# %%
X

# %%
arvore.fit(X, y)
# %%
arvore.predict([[0, 0, 0, 1]])
# %%
import matplotlib.pyplot as plt
plt.figure(dpi=400)
tree.plot_tree(
    arvore, 
    feature_names=caracteristicas,
    class_names=arvore.classes_,
    filled=True)
# %%
proba = arvore.predict_proba([[0,0,0,1]])[0]
pd.Series(proba, index=arvore.classes_)

# %%

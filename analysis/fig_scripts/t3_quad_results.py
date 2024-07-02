import numpy as np
import pandas as pd

df = pd.read_csv("../data/quad_results_table_reps.csv", index_col = 0)
grouped = df.groupby([r'$\sigma$', r'$n$']).agg(
    {r'Time (seconds)': 'mean', 'Error': 'mean'})
# remove the multiindex
grouped.reset_index(inplace=True)
grouped.loc[grouped[r"$\sigma$"].diff() == 0, r"$\sigma$"] = ""
grouped


file = "../../figures/quad_results_table_5day.tex"
with open(file, "w") as f:
    f.write(grouped.to_latex(index = False, multicolumn_format="l", escape=False))

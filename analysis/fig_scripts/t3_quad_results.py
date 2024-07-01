import iris
import numpy as np
import pandas as pd

cubes_dir = "/user/work/hg20831/postproc-share/quadrature/"
n_intervals = [1, 2, 4, 6, 12, 24]
sds = [0.5, 1, 2]
reps = 10
df = pd.DataFrame([(sd, n, rep+1) for sd in sds for n in n_intervals for rep in range(reps)], columns=[r'$\sigma$', r'$n$', r'$m$'])
df['Time (seconds)'] = np.nan
df['Error'] = np.nan

n_rows = len(n_intervals * reps)
for i, sd in enumerate(sds):
    for j, n_int in enumerate(n_intervals):
        for rep in range(reps):
            file_name = cubes_dir + f"quad_exc_prob_{n_int}_sd{str(sd).replace('.', '_')}_5day_{rep}.nc"
            try:
                cube = iris.load(file_name)[0]
            except OSError:
                print(f"{i} {j} {rep} failed.")
                continue
            # Compute 'time' and 'error' here
            time = cube.coord("eval_time").points[0] 
            error = cube.coord("error").points[0]  
            # Update the DataFrame
            df.loc[n_rows*i+j, 'Error'] = error.round(2)
            df.loc[n_rows*i+j, 'Time (seconds)'] = int(time)

print(df)
df.to_csv("quad_results_table_reps.csv")


df = pd.read_csv("quad_results_table_reps.csv", index_col = 0)
grouped = df.groupby([r'$\sigma$', r'$n$']).agg(
    {r'Time (seconds)': 'mean', 'Error': 'mean'})
# remove the multiindex
grouped.reset_index(inplace=True)
grouped.loc[grouped[r"$\sigma$"].diff() == 0, r"$\sigma$"] = ""
grouped


file = "../data/quad_results_table_5day.tex"
with open(file, "w") as f:
    f.write(grouped.to_latex(index = False, multicolumn_format="l", escape=False))

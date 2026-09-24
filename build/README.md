# Building the notebooks

The notebooks in `../notebooks/` are generated, not edited by hand. Each `nbNN.py` file declares the cells of one notebook as Python strings (`md(...)` for markdown, `code(...)` for code, `exercise`/`solution` for the exercise blocks). Running it executes the notebook from top to bottom, writes the result with outputs to `../notebooks/`, and then separately executes the code hidden in every solution block so that a broken solution fails the build.

```
pip install -r ../requirements.txt nbformat nbclient ipykernel
python build_all.py          # rebuild and validate everything (a few minutes)
python nb05.py               # rebuild one notebook
python validate.py           # check the committed notebooks
python fetch_data.py         # refresh ../data/ from the original online sources
```

During a build the data files are read from the local `../data/` folder; the notebooks that are written out refer to the raw-GitHub URLs so that they work in Colab.

`validate.py` checks that every notebook is valid, has its own Colab badge, is linked from the top-level README, contains no error outputs or leaked local paths, and has one solution block per exercise.

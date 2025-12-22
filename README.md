# ScopeBO

A scope selection tool for organic chemistry that balances scope performance and substrate similarity. Check out [our preprint](https://chemrxiv.org/engage/chemrxiv/article-details/6940776cbc44e47cf412790c) for more information!

* Get scope substrate suggestions

* Generate molecular features and the search space input for the algorithm

* Visualize the scope selections on a UMAP

* SHAP feature analysis

* Predictive modeling for unseen substrates outside of the scope

---

### Installation:

(1) Install [Anaconda or Miniconda](https://www.anaconda.com/download/success "Download link").

(2) Download this repository.

(3) Open the terminal in the ScopeBO folder (containing the file "environment.yml").

(4) Run the following command to install the environment:

```
conda env create -f environment.yml
```


(5) Activate the environment by running the following command:

```
conda activate scope_bo
```

(6) Launch Juypter Notebook by running the following command:

```
jupyter notebook
```

Open the file "ScopeBO_example.ipynb" in the folder "Examples" to see a usage example.

See the file "installation_instructions.txt" for more detailed instructions.

---

### Notes

* See the supporting information of our publication for detailed recommendations regarding search space curation, featurization, etc.

* The folder "Data" contains all publication data.


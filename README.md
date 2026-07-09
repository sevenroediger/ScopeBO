# ScopeBO

A scope selection tool for organic chemistry that balances scope performance and substrate similarity. Check out [our preprint](https://chemrxiv.org/engage/chemrxiv/article-details/6940776cbc44e47cf412790c) for more information!

* Get scope substrate suggestions

* Generate molecular features and the search space input for the algorithm

* Visualize the scope selections on a UMAP

* SHAP feature analysis

* Predictive modeling for unseen substrates outside of the scope

* **ScopeBO is also available as an app (see below)**

---

### Installation:

(1) Install [Miniconda](https://www.anaconda.com/download/success "Download link").

*Note*: We noticed that Anaconda can sometimes lead to installation problems due to using a different environment solver.

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

Open the file [ScopeBO_example.ipynb](https://github.com/doyle-lab-ucla/ScopeBO/blob/main/Examples/ScopeBO_example.ipynb) in the folder "Examples" to see a usage example.

See the file [installation-instructions.txt](https://github.com/doyle-lab-ucla/ScopeBO/blob/main/installation_instructions.txt) for more detailed instructions.

---

### App setup:

(1) Before first use: Follow the installation instructions steps 1–4 (see above).

(2) Double-click the app (ScopeBO_App.zip on MacOS or ScopeBO_App.bat on Windows).

*Note*: Unpack the zip folder on MacOS by double-clicking and then open the unpacked ScopeBO_App.app.

*Note*: See the file [installation-instructions.txt](https://github.com/doyle-lab-ucla/ScopeBO/blob/main/installation_instructions.txt) for information on opening the app via the terminal or in case of security settings blocking the opening of the app.


---

### Notes

* See the supporting information of our publication for detailed recommendations regarding search space curation, featurization, etc. This information is also provided in the app via the "?" help buttons.

* Publication data for our manuscript can be found [here](https://github.com/doyle-lab-ucla/ScopeBO_PublicationData).

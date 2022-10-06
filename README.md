## hls4ml_pyg_demo
Walkthrough of current progress of hls4ml pyg for tau3mu CMS experiment

## Installation
We need to setup a conda environment to install all the necessary packages. To install the conda environment, first go to this project's root directory and do: <br />

```console
$ conda env create -f environment.yml
$ conda activate pyg_to_hls_walkthrough
```
<br />

This will take some time to run, but at the end of the end of the day. We should see "(pyg_to_hls_walkthrough)" on the left side of our terminal. This shows that we are in that coding environment.

<br />
Next, we need to replace the normal hls4ml package with the local folder with the same name("hls4ml"). This is because the official package doesn't support pytorch geometric(pyg) conversion while our local directory does.<br />
To find your hls4ml package, go first into your python script and do:<br />

```console
>>> import torch
>>> print(torch.__file__)
/home/swissman777/anaconda3/envs/pyg_to_hls_walkthrough/lib/python3.7/site-packages/torch/__init__.py
```

As we can see, for me, the print function returns "/home/swissman777/anaconda3/envs/pyg_to_hls_walkthrough/lib/python3.7/site-packages/torch/__init__.py". <br />
From this, we go two steps back to find the location where my pip packages are installed: "/home/swissman777/anaconda3/envs/pyg_to_hls_walkthrough/lib/python3.7/site-packages/". <br />
 <br />
All we have to do now is to go to that file location, find the directory named "hls4ml" and replace it with local "hls4ml" directory. Now pyg to hls conversion should be supported.<br />
<br />

## Running the walkthrough
The actual walkthrough is on pyg_to_hls_walkthrough.ipynb. To open it, we first need to open jupyter notebook. To do that, we write in our conda pyg_to_hls_walkthrough terminal:

```console
$ jupyter notebook
```

This should automatically open a browser where you can click "pyg_to_hls_walkthrough.ipynb" to open it. <br />
If it doesn't automatically open, the terminal should print out a link for us to copy and paste onto a browser of our choice just like pasting a url of a website. <br />
When you have opened the notebook, please follow the instructions there. Thank you!



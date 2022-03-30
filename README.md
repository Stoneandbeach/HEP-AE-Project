# HEP-AE-Project
Repository of code used as part of my master's thesis project; Autoencoder Compression in High Energy Physics. To understand this repository, I suggest first reading the report, which is available online here:

https://lup.lub.lu.se/student-papers/search/publication/9075881

The scripts of this repository have grown and changes with the needs of the thesis project. They are not meant to be generalizable, and build upon the specific input file structures, AE architectures, information storage and specific versions of that stored information that was used the thesis.

This whole repository should be seen mainly as a complement to the thesis report. For the most part, it is not really important *how* the scripts work, but rather what the results show, as in the case of the visualizations.

It is, however, important to understand one of the scripts in particular: grouped-encoding-31dim.ipynb. This is an almost entirely self-contained notebook that trains an AE to compress groups of 31D jets from a .pkl file (produced by the data18-root-extraction.ipynb notebook). Similarly, the https://github.com/Stoneandbeach/ATLAS-collective-AE repository contains a notebook that trains an AE to first compress 4D jets to 3D, and then group and compress those 3D latent space representations further.

Those two processes demonstrate what the collective-multi-trainer.py script does, but in a more scaled-down version. The collective-multi-trainer.py script is what was finally used in the thesis for training the large number of AEs used in the report.

Again, though it is possible, this repository is not meant to be used by anyone else. If you want to use or adapt my code for another project, please feel free to contact me with any questions!

The versions of main python libraries used in my thesis project are:

numpy 1.19.2<br>
pandas 1.2.3<br>
pytorch 1.8.1<br>
fastai 2.3.1<br>
vector 0.8.1<br>
uproot 4.0.7<br>

The complete list of libraries in the Anaconda environment used is found in the file environment.txt.

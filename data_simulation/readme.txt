
The code in here started in the simulated single cell benchmark repo, and I lifted it over when I wanted to start changing simulation parameters and such in this project. It makes sense to have that code here, and I am working towards one repo with all or most of the code for grn inference, regardless of project, to avoid doing the same work multiple times.

Because this code bridges Matlab and Python, and I want to do minimal stuff in Matlab, it follows the structure: 
- Configure simulations in Python
- Do the simulation job handling (multiprocessing) in Python
- Run matlab jobs via subprocess
- Matlab reads all parameters (possible) from the configurations
- To avoid synchronising the output of the multiprocessing in matlab, matlab writes to files. The filenames are included in the configurations
- Python reads simulation outputs files and compiles results

It has at least two limitations:
- The synchronisation problem is of course not solved, just deferred the the filesystem (but with configs that specify non overlapping paths). Thus, if running too many matlab jobs in parallel, the work becomes IO bound
- matlab is naughty when it comes to multiprocessing. At least during startup, each instance does it's own internal multiprocessing, even when run in batch mode, and even if the script being run has settings to run single threaded. Thus, if running too many matlabs in parallel, they seem to collide during matlab startup. Matlab is fundamentally and IDE with a language built into it. Unfortunately.

Bespite those downsides, with reasonable (probably current) multiprocessing parameters, it's working pretty well. And it is handling the bridging of Python and Matlab. Thus not planning to change it unless I have to, according to the idea of not fixing what isn't broken.

And now you hopefully understand the overall structure of the code in here.




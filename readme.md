
Anton Björks GRN related code

Previously had different repositories per project. Plan is to have one repository for GRN related work, to avoid repeating work, since most GRN projects do similar things.


There is a distinction between code that is meant to be reusable across different analyses, and code that is specific to a particular one.
analyses/
contains code specific to analyses.
src/
contains code meant to reuse. The Python packaging behavior aligns with this. Things in src/ are contained in the package, and so can be robustly imported.



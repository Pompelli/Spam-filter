I decided for a Bayesscher Spamfilter
https://de.wikipedia.org/wiki/Bayesscher_Spamfilter

The file "main.py" contains my code with detailed descriptions.

The file "test.py" was used solely to determine the best parameters for my classifier through grid search. The optimal parameters were then applied in the main file. Since "test.py" is not part of the submission and not required, it has not been commented. I included it only to demonstrate how the chosen hyperparameters were determined.



to Run:

pdm install
pdm run main.py
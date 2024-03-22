# Pyvectorial
This is a library for interfacing with vectorial models of cometary atmospheres, namely the model included in sbpy, but other versions written in fortran and rust are supported also.
Features include standardized model input and output, database caching of completed models, and parallel running of models.

# Testing
`pytest`

## Coverage Report
`pytest --cov-report term-missing --cov=pyvectorial src/pyvectorial/tests`

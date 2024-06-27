# Code Source
This folder contains the code for the thesis; the test, benchmark, model, and heuristic code.

## ACO - Ant Colony Optimisation
For a minimal version of the ACO see the code below.
```python
from src.production_orders import parse_data
from src.schedule_generator.main import JobShopProblem, ObjectiveFunction
from src.schedule_generator.ant_colony_optimisation import TwoStageACO

# Load the problem
jssp = JobShopProblem.from_data(parse_data("path/to/excel/sheet"))

# Instantiate the ACO with the objective of minimising the makespan
aco = TwoStageACO(jssp, ObjectiveFunction.MAKESPAN)

# Run the Ant Colony for some generations
aco.run()

# Optional - Visualise the schedule
schedule = jssp.make_schedule_from_parallel(aco.best_solution[1])
jssp.visualize(schedule)
```

The ACO has some parameters that are important to consider when running it in order to achieve the best results. Some advice is to use the parallel with stock schedule generator since that one generally gives better solution. Furthermore, setting `tau_zero` as a rough estimate for the objective function is important so that the model does not have too small float numbers, and also to achieve better results. Increasing the number of ants does not necessarily provide better results, as there is a diminishing return. Lastly setting the convergence rate is important as to not land in a local optimal. Depending on the problem it could be interesting to use local search, however for the provided data sets this has not proven to be useful.

> [!NOTE]
> The `JobShopProblem.from_data()` method is unique for this problem that has been developed. If you wish to extend this model to a new problem that does not have the same constraints I would recommend to create a new class. You should inherit from the JobShopProblem, but change the functions so that it fits your problem. Once that is done it will seamlessly fit into the `TwoStageACO` class, and can be solved. Please make sure that the return types are similar.

## Structure
* `src/examples/` contains three data sets for the thesis. These are small (5 orders), medium (7 orders), and large (100 orders) in size.
* `src/schedule_generator/` contains code for loading a `JobShopProblem` instance and the code for the Ant Colony Optimisation.
* `src/production_orders.py` contains the code for loading the data sets into `pydantic` and pandas DataFrame objects.
* `src/tests/` contains tests for the `production_orders.py` code.

## Problems
### Numba Error
If you get an error regarding numba it is likely that your computer cannot run it. Thus, to avoid this error you can comment out the function decorator in `src/schedule_generator/numba_numpy_functions.py`, so that it does not precompile that function. This means it will run a bit more slow, since the function will not be compiled into machine code through numba.
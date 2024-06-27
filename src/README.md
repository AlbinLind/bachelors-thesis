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

Since some systems may have problems pre-compiling numpy functions with the help of numba it is avoided by default. If you wish to speed up the code uncomment the code in `src/schedule_generator/numba_numpy_functions.py`. This is an optional option, but it does improve the runtime significantly.

Depending on how verbose you may want the heuristic to be you can specify two different parameters.
* `verbose`: set to true if you want the model to print more often, this includes information about each generation and if a new best solution has been found.
* `quite`: set to true if you do not want any information printed to stdout at all. This is usefull for benchmarks for example.

> [!NOTE]
> The `JobShopProblem.from_data()` method is unique for this problem that has been developed. If you wish to extend this model to a new problem that does not have the same constraints I would recommend to create a new class. You should inherit from the JobShopProblem, but change the functions so that it fits your problem. Once that is done it will seamlessly fit into the `TwoStageACO` class, and can be solved. Please make sure that the return types are similar.

## Exact Method
The exact method resides in `src/schedule_generator/exact_solution.py`. This model struggles with bigger magnitudes of production orders, and are only really able to solve problems up to around 7 produciton orders. In order to solve it its highly recomended to have access to a strong solver such as CPLEX (which is also the only officially tested solver), since the mathematical model is rather complex. Below is an example on how it could be solved with CPLEX installed on your system.

```python
from src.schedule_generator.main import JobShopProblem
from src.production_orders import parse_data
from src.exact_solution import generate_model, solve_model, get_schedule

# Load the problem
jssp = JobShopProblem.from_problem(parse_data("path/to/data"))

# Generate a pyomo concrete model
model = generate_model(jssp)

# Solve the model with CPLEX, you can also provide a time limit
solve_model(model)

# (Optional) visualize the schedule
sc = get_schedule(model, jssp)
jssp.visualize_schedule(sc)
```

## Structure
* `src/examples/` contains three data sets for the thesis. These are small (5 orders), medium (7 orders), and large (100 orders) in size.
* `src/schedule_generator/` contains code for loading a `JobShopProblem` instance and the code for the Ant Colony Optimisation.
* `src/production_orders.py` contains the code for loading the data sets into `pydantic` and pandas DataFrame objects.
* `src/tests/` contains tests for the `production_orders.py` code.

## Problems
### Numba Error
If you get an error regarding numba it is likely that your computer cannot run it. Thus, to avoid this error you can comment out the function decorator in `src/schedule_generator/numba_numpy_functions.py`, so that it does not precompile that function. This means it will run a bit more slow, since the function will not be compiled into machine code through numba.
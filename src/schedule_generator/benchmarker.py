import time

import pandas as pd
from src.schedule_generator.ant_colony_optimisation import TwoStageACO
from src.schedule_generator.main import JobShopProblem, ObjectiveFunction
from src.production_orders import Data, parse_data


def run_benchmark_custom_objectives():
    small_data = parse_data("examples/data_v1_small.xlsx")
    medium_data = parse_data("examples/data_v1.xlsx")
    large_data = parse_data("examples/data_v1_large.xlsx")
    data_sets: dict[str, Data] = {
        "small": small_data,
        "medium": medium_data,
        "large": large_data,
    }
    seeds = [46589, 2034, 1320, 65490834, 73485]
    info = {
        "data_size": [],
        "runtime": [],
        "makespan": [],
        "total_setup_time": [],
        "classical_tardiness": [],
        "objective_function": [],
        "solution": [],
        "seed": [],
    }
    lower_values = {
        "small": {"makespan": 2279.0, "total_setup_time": 35.0, "tardiness": 0.1},
        "medium": {"makespan": 3499.0, "total_setup_time": 95.0, "tardiness": 8.0},
        "large": {"makespan": 16000.0, "total_setup_time": 1600.0, "tardiness": 650.0},
    }
    info = {
        "data_size": [],
        "runtime": [],
        "makespan": [],
        "total_setup_time": [],
        "classical_tardiness": [],
        "boolean_tardiness": [],
        "custom_objective": [],
        "solution": [],
        "seed": [],
    }
    for data_size, data in data_sets.items():
        jssp = JobShopProblem.from_data(data)
        jssp.LOW_MAKESPAN = lower_values[data_size]["makespan"]
        jssp.LOW_TOTAL_SETUP_TIME = lower_values[data_size]["total_setup_time"]
        jssp.LOW_TARDINESS = lower_values[data_size]["tardiness"]
        for seed in seeds:
            print(
                f"Running benchmark {len(info['runtime']) + 1}/15 completed: {len(info['runtime'])/15*100:.2f}%"
            )
            aco = TwoStageACO(
                jssp,
                ObjectiveFunction.CUSTOM_OBJECTIVE,
                seed=seed,
                tau_zero=1 / 0.1,
                quite=True,
                n_ants=10,
                n_iter=5_000,
                rho=0.01,
                time_limit=60 * 5,
                with_local_search=False,
                with_stock_schedule=True,
            )
            start_time = time.time()
            aco.run()
            end_time = time.time()
            info["data_size"].append(data_size)
            info["runtime"].append(end_time - start_time)
            schedule = aco.problem.make_schedule_from_parallel_with_stock(
                aco.best_solution[1]
            )
            info["makespan"].append(aco.problem.makespan(schedule))
            info["total_setup_time"].append(aco.problem.total_setup_time(schedule))
            info["classical_tardiness"].append(
                aco.problem.classical_tardiness(schedule)
            )
            info["boolean_tardiness"].append(aco.problem.boolean_tardiness(schedule))
            info["custom_objective"].append(aco.problem.custom_objective(schedule))
            info["solution"].append(aco.best_solution[1])
            info["seed"].append(seed)
    pd.DataFrame(info).to_csv("benchmark_results_eta_with_production.csv", index=False)


def run_benchmark_simple_objectives():
    small_data = parse_data("examples/data_v1_small.xlsx")
    medium_data = parse_data("examples/data_v1.xlsx")
    large_data = parse_data("examples/data_v1_large.xlsx")
    data_sets = {"small": small_data, "medium": medium_data, "large": large_data}
    objective_functions = {
        "makespan": ObjectiveFunction.MAKESPAN,
        "classical_tardiness": ObjectiveFunction.CLASSICAL_TARDINESS,
    }
    seeds = [234, 2458690, 234509, 456799852, 638450]
    info = {
        "data_size": [],
        "runtime": [],
        "makespan": [],
        "total_setup_time": [],
        "classical_tardiness": [],
        "objective_function": [],
        "solution": [],
        "seed": [],
    }
    number_of_iterations = len(data_sets) * len(objective_functions) * len(seeds)
    complete_time = time.time()
    try:
        for data_size, data in data_sets.items():
            jssp = JobShopProblem.from_data(data)
            for (
                objective_function_name,
                objective_function,
            ) in objective_functions.items():
                for seed in seeds:
                    print(
                        f"Running benchmark {len(info['runtime']) + 1}/{number_of_iterations} completed: {len(info['runtime'])/number_of_iterations*100:.2f}%"
                    )
                    if len(info["runtime"]) > 0:
                        # Estimate the time left
                        current_time = time.time()
                        time_per_iteration = (current_time - complete_time) / len(
                            info["runtime"]
                        )
                        print(
                            f"Estimated time left: {time_per_iteration * (number_of_iterations - len(info['runtime'])) / 60:.2f} minutes, average time per iteration: {time_per_iteration:.2f} seconds"
                        )
                    aco = TwoStageACO(
                        jssp,
                        objective_function,
                        n_ants=10,
                        n_iter=5_000,
                        rho=0.01,
                        with_stock_schedule=True,
                        with_local_search=False,
                        time_limit=60 * 5,
                        seed=seed,
                        quite=True,
                    )
                    start_time = time.time()
                    aco.run()
                    end_time = time.time()
                    info["data_size"].append(data_size)
                    info["runtime"].append(end_time - start_time)
                    schedule = aco.problem.make_schedule_from_parallel_with_stock(
                        aco.best_solution[1]
                    )
                    info["makespan"].append(aco.problem.makespan(schedule))
                    info["total_setup_time"].append(
                        aco.problem.total_setup_time(schedule)
                    )
                    info["classical_tardiness"].append(
                        aco.problem.classical_tardiness(schedule)
                    )
                    info["objective_function"].append(objective_function_name)
                    info["solution"].append(aco.best_solution[1])
                    info["seed"].append(seed)
    except Exception as e:
        print(f"An error occurred: {e}\nSaving the results to a file...")
    except KeyboardInterrupt:
        print("The benchmark was interrupted. Saving the results to a file...")
        # Make sure all the lists are the same length
        for key in info:
            info[key] = info[key][: len(info["data_size"])]
    pd.DataFrame(info).to_csv("benchmark_results.csv", index=False)


def run_benchmark_local_search():
    small_data = parse_data("examples/data_v1_small.xlsx")
    medium_data = parse_data("examples/data_v1.xlsx")
    large_data = parse_data("examples/data_v1_large.xlsx")
    # data_sets = {"small": small_data, "medium": medium_data}
    data_sets = {"large": large_data}
    objective_functions = {
        "makespan": ObjectiveFunction.MAKESPAN,
        # "classical_tardiness": ObjectiveFunction.CLASSICAL_TARDINESS,
    }
    seeds = [12, 809, 234534850]
    info = {
        "data_size": [],
        "runtime": [],
        "makespan": [],
        "total_setup_time": [],
        "classical_tardiness": [],
        "objective_function": [],
        "solution": [],
        "seed": [],
        "local_search": [],
    }
    number_of_iterations = len(data_sets) * len(objective_functions) * len(seeds) * 2
    complete_time = time.time()
    try:
        for data_size, data in data_sets.items():
            jssp = JobShopProblem.from_data(data)
            for (
                objective_function_name,
                objective_function,
            ) in objective_functions.items():
                for seed in seeds:
                    for local_search in [True, False]:
                        print(
                            f"Running benchmark {len(info['runtime']) + 1}/{number_of_iterations} completed: {len(info['runtime'])/number_of_iterations*100:.2f}%"
                        )
                        if len(info["runtime"]) > 0:
                            # Estimate the time left
                            current_time = time.time()
                            time_per_iteration = (current_time - complete_time) / len(
                                info["runtime"]
                            )
                            print(
                                f"Estimated time left: {time_per_iteration * (number_of_iterations - len(info['runtime'])) / 60:.2f} minutes, average time per iteration: {time_per_iteration:.2f} seconds"
                            )
                        aco = TwoStageACO(
                            jssp,
                            objective_function,
                            n_ants=10,
                            n_iter=5_000,
                            rho=0.01,
                            with_stock_schedule=True,
                            with_local_search=local_search,
                            local_search_iterations=20,
                            time_limit=60 * 5,
                            seed=seed,
                            quite=True,
                        )
                        start_time = time.time()
                        aco.run()
                        end_time = time.time()
                        info["data_size"].append(data_size)
                        info["runtime"].append(end_time - start_time)
                        schedule = aco.problem.make_schedule_from_parallel_with_stock(
                            aco.best_solution[1]
                        )
                        info["makespan"].append(aco.problem.makespan(schedule))
                        info["total_setup_time"].append(
                            aco.problem.total_setup_time(schedule)
                        )
                        info["classical_tardiness"].append(
                            aco.problem.classical_tardiness(schedule)
                        )
                        info["objective_function"].append(objective_function_name)
                        info["solution"].append(aco.best_solution[1])
                        info["seed"].append(seed)
                        info["local_search"].append(local_search)
    except Exception as e:
        print(f"An error occurred: {e}\nSaving the results to a file...")
    except KeyboardInterrupt:
        print("The benchmark was interrupted. Saving the results to a file...")
        # Make sure all the lists are the same length
        for key in info:
            info[key] = info[key][: len(info["data_size"])]
    pd.DataFrame(info).to_csv("benchmark_results_local_search_large.csv", index=False)


def run_benchmark_normal_schedule():
    medium_data = parse_data("examples/data_v1.xlsx")
    aco = TwoStageACO(
        JobShopProblem.from_data(medium_data),
        objective_function=ObjectiveFunction.MAKESPAN,
        n_ants=100,
        tau_zero= 1.0 / 3500,
        n_iter=5000,
        seed=100,
        with_local_search=True,
        verbose=True,
    )
    aco.run()
    print(aco.best_solution[0])


if __name__ == "__main__":
    # run_benchmark()
    # run_benchmark_custom_objectives()
    # run_benchmark_local_search()
    run_benchmark_normal_schedule()

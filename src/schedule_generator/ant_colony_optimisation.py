import time
from typing import Self

from src.production_orders import parse_data
from src.schedule_generator.main import JobShopProblem, ObjectiveFunction
from src.schedule_generator.numba_numpy_functions import select_random_item, nb_set_seed
import numpy as np


class TwoStageACO:
    def __init__(
        self,
        problem: JobShopProblem,
        objective_function: ObjectiveFunction,
        *,
        n_ants: int = 500,
        n_iter: int = 50,
        time_limit: int | None = None,
        seed: int = 42,
        rho: float = 0.1,
        alpha: float = 0.1,
        beta: float = 2.0,
        q_zero: float = 0.9,
        tau_zero: float = 1.0,
        verbose: bool = False,
        quite: bool = False,
        with_stock_schedule: bool = False,
        with_local_search: bool = True,
        local_search_iterations: int = 20,
        convergence_limit: int = 20,
    ) -> None:
        """Initializes the two-stage ACO algorithm.

        To run the algorithm you may either run your own loop with generation logic, then
        it is adviced to use the `run_and_update_ant` method. However, it is suggested to
        use the `run` method which will run the algorithm for `n_iter` iterations, or until
        the time limit is reached (if one is set), whichever comes first.

        Args:
            problem (JobShopProblem): the job shop problem to solve.
            objective_function (ObjectiveFunction): the objective function to use.
            n_ants (int, optional): number of ants to use, a good indication is (# of jobs)/2 based on
                [this](https://arxiv.org/ftp/arxiv/papers/1309/1309.5110.pdf). Defaults to 10.
            n_iter (int, optional): number of iterations (i.e. how many times global update will happen).
                Defaults to 50.
            time_limit (int | None, optional): time limit in seconds, if None then no time limit is set. Defaults to None.
            seed (int, optional): seed for numpy.random. Defaults to 42.
            rho (float, optional): parameter for local update, and how much pheromones we evaoprate. Defaults to 0.1.
            alpha (float, optional): paramter for global update, and how much the pheromones evaporate. Defaults to 0.1.
            beta (float, optional): parameter for the weight we put on the heuristic value. Defaults to 2.0.
            q_zero (float, optional): paramter for how often we just choose the highest probability (exploitation). Defaults to 0.9.
            tau_zero (float, optional): parameter for normalisation of pheromones levels, a good estimate for this is 1.0/(Z_best)
                where Z_best is a rough approximation of the optimal objective value. Defaults to 1.0.
            verbose (bool, optional): defines how much should be printed to stdout. Defaults to False.
            quite (bool, optional): defines if the algorithm should not print anything to stdout. Defaults to False.
            with_stock_schedule (bool, optional): defines if the stock schedule should be used. Defaults to False.
            with_local_search (bool, optional): defines if local search should be used. Defaults to True.
            local_search_iterations (int, optional): number of iterations for the local search. Defaults to 20.
            convergence_limit (int, optional): number of iterations with the same best solution before the pheromones are reset. Defaults to 20.
        """
        self.problem = problem
        self.n_ants = n_ants
        self.n_iter = n_iter
        np.random.seed(seed)
        nb_set_seed(seed)
        self.rho = rho / self.n_ants
        self.alpha = alpha
        self.beta = beta
        self.q_zero = q_zero
        self.tau_zero = tau_zero
        self.verbose = verbose
        self.objective_function = objective_function

        # pheromones will be (tasks + 1) x (tasks + 1) matrix, since there is
        # a fictional starting node.
        self.pheromones_stage_one = np.ones(
            (len(self.problem.jobs), len(self.problem.machines))
        )
        self.pheromones_stage_two = np.ones(
            (
                len(self.problem.jobs) + 1,
                len(self.problem.jobs) + 1,
                len(self.problem.machines),
            )
        )
        # Initialize the best solution so far
        self.with_stock_schedule = with_stock_schedule
        self.with_local_search = with_local_search
        self.local_search_iterations = local_search_iterations
        self.conv_limit = convergence_limit
        self.time_limit = time_limit
        self.quite = quite
        self.best_solution: tuple[float, np.ndarray] = (1e100, np.zeros((1, 1)))

    def evaluate(self, parallel_schedule: np.ndarray) -> float:
        """Evaluates the path and machine assignment. This function takes job order based on the machines
        and evaluates the objective function."""
        if self.with_stock_schedule:
            schedule = self.problem.make_schedule_from_parallel_with_stock(
                parallel_schedule
            )
        else:
            schedule = self.problem.make_schedule_from_parallel(parallel_schedule)
        if self.objective_function == ObjectiveFunction.MAKESPAN:
            return self.problem.makespan(schedule)
        elif self.objective_function == ObjectiveFunction.TARDINESS:
            return self.problem.tardiness(schedule)
        elif self.objective_function == ObjectiveFunction.TOTAL_SETUP_TIME:
            return self.problem.total_setup_time(schedule)
        elif self.objective_function == ObjectiveFunction.CUSTOM_OBJECTIVE:
            return self.problem.custom_objective(schedule)
        elif self.objective_function == ObjectiveFunction.BOOLEAN_TARDINESS:
            return self.problem.boolean_tardiness(schedule)
        elif self.objective_function == ObjectiveFunction.CLASSICAL_TARDINESS:
            return self.problem.classical_tardiness(schedule)
        else:
            raise ValueError(
                f"Objective function {self.objective_function} not supported."
            )

    def assign_machines(self) -> dict[int, set[int]]:
        """Assigns machines to jobs based on the pheromones and the heuristic value.

        Returns:
            dict[int, set[int]]: the machine assignment for each job, where the key is the machine index
                and the values is a set of job indices.
        """
        assignment: dict[int, set[int]] = {
            machine: set() for machine in range(len(self.problem.machines))
        }
        for idx, job in enumerate(self.problem.jobs):
            available_machines = list(job.available_machines.keys())
            if len(available_machines) == 1:
                assignment[available_machines[0]].add(idx)
                continue
            probabilities = np.zeros(len(available_machines))
            denominator = 0.0
            for machine_idx, machine in enumerate(available_machines):
                tau_r_s = self.pheromones_stage_one[idx, machine]
                eta_r_s = 1.0 / job.available_machines[machine]
                numerator = tau_r_s * eta_r_s**self.beta
                probabilities[machine_idx] = numerator
                denominator += numerator

            # Avoid division by zero when we have very small numbers
            if denominator <= 1e-6:
                chosen_machine = select_random_item(available_machines)
                assignment[chosen_machine].add(idx)
                continue
            probabilities = probabilities / denominator
            chosen_machine = select_random_item(
                available_machines, probabilities=probabilities
            )
            assignment[chosen_machine].add(idx)
        return assignment

    def global_update_pheromones(self):
        """Globally updates the pheromones based on the best solution found so far (elitistic)."""
        inverse_best_value = (1.0 / self.best_solution[0]) / self.tau_zero
        self.pheromones_stage_one *= 1 - self.alpha
        self.pheromones_stage_two *= 1 - self.alpha
        for m_idx, order in enumerate(self.best_solution[1]):
            for idx, job_idx in enumerate(order):
                # If the job index is -2 then we have reached the end of the schedule.
                if job_idx == -2:
                    break
                # If the index is 0 then we are at the start of the schedule, and thus at the ficticious job.
                if idx == 0:
                    continue
                # Update stage two
                last_job_idx = order[idx - 1]
                self.pheromones_stage_two[last_job_idx, job_idx, m_idx] += (
                    self.alpha * inverse_best_value
                )
                # If the job index is -1 then we have reached the first ficticious job.
                if job_idx == -1:
                    continue
                # Update stage one
                self.pheromones_stage_one[job_idx, m_idx] += (
                    self.alpha * inverse_best_value
                )

    def local_update_pheromones(self, schedule: np.ndarray):
        """Does a local update of the pheromones, by evaporation.

        Only the paths that are taken by the ants are updated to allow for more exploration.

        Args:
            schedule (np.ndarray): the schedule to update the pheromones based on.
        """
        for machine in range(len(self.problem.machines)):
            for idx, job_idx in enumerate(schedule[machine]):
                if job_idx == -2:
                    break
                if idx == 0:
                    continue
                # Update stage two
                last_job_idx = schedule[machine][idx - 1]
                self.pheromones_stage_two[last_job_idx, job_idx, machine] = (
                    self.pheromones_stage_two[last_job_idx, job_idx, machine]
                    * (1 - self.rho)
                )
                if job_idx == -1:
                    continue
                # Update stage one
                self.pheromones_stage_one[job_idx, machine] = self.pheromones_stage_one[
                    job_idx, machine
                ] * (1 - self.rho)

    def draw_job_to_schedule(
        self, jobs_to_schedule: set[int], last: int, machine: int
    ) -> int:
        """Randomly draws a job to schedule based on the pheromones and the heuristic value.

        Args:
            jobs_to_schedule (set[int]): jobs that are available to schedule.
            last (int): the last job that was scheduled.
            machine (int): the machine that the job should be scheduled on, which has been designated earlier.

        Returns:
            int: the job index that was drawn to be scheduled.
        """
        jobs_to_schedule_list = list(jobs_to_schedule)

        probabilites = np.zeros(len(jobs_to_schedule_list))
        denominator = 0.0
        for idx, job in enumerate(jobs_to_schedule_list):
            tau_r_s = self.pheromones_stage_two[last, job, machine]
            eta_r_s = 1.0 / (1 + self.problem.setup_times[last, job])
            numerator = tau_r_s * eta_r_s**self.beta
            probabilites[idx] = numerator
            denominator += numerator

        # The pseudo-random number is used to determine if we should exploit or explore.
        if np.random.rand() <= self.q_zero:
            return jobs_to_schedule_list[np.argmax(probabilites)]
        # Avoid division by zero when we have very small numbers
        if denominator <= 1e-6:
            return select_random_item(jobs_to_schedule_list)
        probabilites = probabilites / denominator
        return select_random_item(jobs_to_schedule_list, probabilities=probabilites)

    def local_search(
        self,
        schedule: np.ndarray,
        machine_assignment: dict[int, set[int]],
        schedule_objective_value: float,
    ):
        """A simple local search algorithm that swaps a few jobs on the same machine.

        Args:
            schedule (np.ndarray): schedule to perform local search on.
            machine_assignment (dict[int, set[int]]): the machine assignment for each job.
            schedule_objective_value (float): the objective value of the initial schedule.

        Returns:
            np.ndarray: new schedule after local search.
        """
        new_schedule = schedule  # .copy()
        for _ in range(self.local_search_iterations):
            x = np.random.rand()
            operation = (0, 0, 0)
            if x < 1:
                # Swap two jobs on the same machine
                machine = np.random.randint(len(self.problem.machines))
                number_of_jobs_on_machine = len(machine_assignment[machine])
                if number_of_jobs_on_machine < 2:
                    continue
                job1_idx = np.random.randint(1, number_of_jobs_on_machine)
                job2_idx = np.random.randint(1, number_of_jobs_on_machine)
                new_schedule[machine][job1_idx], new_schedule[machine][job2_idx] = (
                    new_schedule[machine][job2_idx],
                    new_schedule[machine][job1_idx],
                )
                operation = (machine, job1_idx, job2_idx)
            objective_value = self.evaluate(new_schedule)
            if objective_value < schedule_objective_value:
                schedule = new_schedule
            else:
                machine, job1_idx, job2_idx = operation
                new_schedule[machine][job1_idx], new_schedule[machine][job2_idx] = (
                    new_schedule[machine][job2_idx],
                    new_schedule[machine][job1_idx],
                )
        return new_schedule

    def run_ant(self) -> tuple[np.ndarray, dict[int, set[int]]]:
        """Run the ant and return the job order and machine assignment.

        Returns:
            tuple[np.ndarray, dict[int, set[int]]]: job order and machine assignment.
        """
        machine_assignment = self.assign_machines()
        # Initialise the schedules with -2, which is the end of the schedule.
        schedules = (
            np.ones(
                (len(self.problem.machines), len(self.problem.jobs)), dtype=np.int32
            )
            * -2
        )
        for machine in range(len(self.problem.machines)):
            schedules[machine, 0] = -1
            jobs_assigned = set()
            for i in range(len(machine_assignment[machine])):
                jobs_to_schedule = machine_assignment[machine].difference(jobs_assigned)

                jobs_to_schedule = {
                    job for job in jobs_to_schedule if set(self.problem.jobs[job].dependencies).intersection(machine_assignment[machine]).issubset(jobs_assigned)
                }

                job_idx = self.draw_job_to_schedule(
                    jobs_to_schedule=jobs_to_schedule,
                    last=schedules[machine][i],
                    machine=machine,
                )
                schedules[machine, i + 1] = job_idx
                jobs_assigned.add(job_idx)
        return schedules, machine_assignment

    def run_and_update_ant(self):
        """Run the ant, that is produce a schedule, and update the pheromones with the local update.

        Returns:
            float: the objective value of the schedule.
        """
        schedule, machine_assignment = self.run_ant()
        objective_value = self.evaluate(schedule)
        if self.with_local_search:
            schedule = self.local_search(schedule, machine_assignment, objective_value)
            objective_value = self.evaluate(schedule)
        self.local_update_pheromones(schedule)
        if objective_value <= self.best_solution[0]:
            self.best_solution = (objective_value, schedule)
            if self.verbose:
                print(f"New best solution: {self.best_solution[0]}")
        return objective_value

    def run(self):
        """Run the algorithm for `n_iter` iterations, or until the time limit is reached (if set).

        This method will in addition to the local update also do the global update of the pheromones.
        """
        min_same = 0
        start_time = time.time()
        for gen in range(self.n_iter):
            results = list()
            for _ in range(self.n_ants):
                results.append(self.run_and_update_ant())
            minimum = np.min(results)
            if minimum == 0:
                print("Optimal solution found stopping...")
                break
            if self.verbose:
                print(
                    f"Generation {gen}, best objective value: {self.best_solution[0]}"
                )
            elif gen % 10 == 0 and not self.quite:
                print(
                    f"Generation {gen}, best objective value: {self.best_solution[0]} "
                    f"max={np.max(results):.3f},min={minimum:.3f},mean={np.mean(results):.3f},std={np.std(results):.3f}"
                )
            self.global_update_pheromones()
            if self.time_limit:
                if time.time() - start_time > self.time_limit:
                    print("Time limit reached, stopping...")
                    break
            if abs(np.min(results) - self.best_solution[0]) < 1e-6:
                min_same += 1
            else:
                min_same = 0
            if min_same == self.conv_limit:
                if self.verbose:
                    print("Resetting pheromones")
                self.pheromones_stage_one = self.pheromones_stage_one * 0 + 1
                self.pheromones_stage_two = self.pheromones_stage_two * 0 + 1
                min_same = 0

    def save(self, name: str):
        """Saves the current ACO instance with its pheromones and best solution to a file.
        It also saves the set tau_zero."""
        np.savez_compressed(
            file=f"{name}.npz",
            stage_one=self.pheromones_stage_one,
            stage_two=self.pheromones_stage_two,
            best_solution_order=self.best_solution[1],
            best_solution_info=np.array(
                [self.best_solution[0], self.objective_function.value, self.tau_zero]
            ),
        )

    @classmethod
    def load(cls, name: str, jssp: JobShopProblem, **kwd) -> Self:
        """Load a previously saved ACO instance."""
        data = np.load(f"{name}.npz")
        aco = cls(
            jssp,
            ObjectiveFunction(data["best_solution_info"][1]),
            tau_zero=data["best_solution_info"][2],
            **kwd,
        )
        aco.pheromones_stage_one = data["stage_one"]
        aco.pheromones_stage_two = data["stage_two"]
        aco.best_solution = (data["best_solution_info"][0], data["best_solution_order"])
        return aco


if __name__ == "__main__":
    data = parse_data(
        "examples/data_v1.xlsx"
    )
    jssp = JobShopProblem.from_data(data)
    aco = TwoStageACO(
        jssp,
        ObjectiveFunction.MAKESPAN,
        verbose=False,
        n_iter=10_000,
        n_ants=10,
        tau_zero=1.0 / (16000.0),
        q_zero=0.9,
        with_stock_schedule=True,
        seed=834,
        with_local_search=False,
        local_search_iterations=30,
        alpha=0.1,
        rho=0.01,
        time_limit=60 * 10,
    )
    # aco = TwoStageACO.load("custom_version_1_0", jssp, n_iter=10000, n_ants = 10, seed=2358002486, with_stock_schedule=True, rho=0.01, with_local_search=False, q_zero=0.95)
    start_time = time.time()
    aco.run()
    print(f"Time taken: {time.time() - start_time}")
    print(aco.best_solution)
    sc = aco.problem.make_schedule_from_parallel_with_stock(aco.best_solution[1])
    print(
        f"custom: {jssp.custom_objective(sc):.3f}, makespan: {jssp.makespan(sc):.3f}, boolean tardiness: {jssp.boolean_tardiness(sc):.3f}, total setup time: {jssp.total_setup_time(sc):.3f}"
    )
    # print(f"{jssp.classical_tardiness(sc)}")
    # aco.problem.visualize_schedule(sc)

    # aco.save("custom_version_1_0")
    # plt.imshow(aco.pheromones_stage_one)
    # plt.colorbar()
    # plt.show()
    # aco.problem.visualize_schedule(
    #     aco.problem.make_schedule_from_parallel(aco.best_solution[1])
    # )

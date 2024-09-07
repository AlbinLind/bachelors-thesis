"""Functions for loading the standard specification of benchmark datasets.

Standard specification
On the first line are two numbers, the first is the number of jobs and the second
the number of machines. Following there is a line for each job. The order for visiting
the machines is presented together with the corresponding processing time. The
numbering of the machines starts at 0.

For example an instance with only a single job on three machines where the processing
time is 5 on machine 1, 6 on machine 2 and 7 on machine 3 and the order that the machines
are to be visited by that job is 2,3,1. The instance would be presented as:

1	3
1	6	2	7	0	5
"""

import matplotlib.pyplot as plt
from platform import machine
from numpy import ndarray
import numpy as np
from src.production_orders import Data
from src.schedule_generator.ant_colony_optimisation import TwoStageACO
from src.schedule_generator.main import Job, Machine, JobShopProblem, ObjectiveFunction, ScheduleError, schedule_type
from src.schedule_generator.numba_numpy_functions import select_random_item

class SimpleJobShopProblem(JobShopProblem):
    def __init__(self, *, data: Data = Data.empty(), jobs: list[Job], machines: list[Machine], number_of_tasks: int = 0) -> None:
        self.number_of_tasks = number_of_tasks
        super().__init__(data=data, jobs=jobs, machines=machines)

    def make_schedule_from_parallel(self, job_orders: list[list[int]] | ndarray) -> dict[int, list[tuple[int, int, int]]]:
        """Make a schedule from a parallel schedule. This function is more simple than the one in the parent class."""
        schedule: dict[int, list[tuple[int, int, int]]] = {
            m.machine_id: [(-1, 0, m.start_time)] for m in self.machines
        }
        job_schedule: dict[int, tuple[int, int, int]] = dict()

        jobs_assigned = set()

        # Choose the job that has the lowest index in the job_orders list, with modulo n_jobs
        current_index = [0 for _ in range(len(self.machines))]
        current_jobs = job_orders[:,current_index].diagonal()

        for idx, j in enumerate(current_jobs):
            if j == -1:
                current_index[idx] += 1
        current_jobs = job_orders[:,current_index].diagonal()

        while not np.all(current_jobs == -2):
            # Get machine index modulo n_jobs
            available_jobs = current_jobs % (self.number_of_tasks)
            available_jobs[current_jobs == -2] = len(self.jobs) + 1
            for idx, j in enumerate(current_jobs):
                if not set(self.jobs[j].dependencies).issubset(jobs_assigned):
                    available_jobs[idx] = len(self.jobs) + 1

            
            # Get the job index with the lowest index
            lowest_indecies = np.where(available_jobs == available_jobs.min())
            lowest_index = lowest_indecies[0][0]
            if len(lowest_indecies[0]) > 1:
                lowest_index = lowest_indecies[0][np.argmin(np.array(current_index)[lowest_indecies[0]])]
                if available_jobs[lowest_index] == len(self.jobs) + 1:
                    raise ScheduleError("No job can be scheduled, because of circular dependencies.")

            machine_idx = int(lowest_index)
            job_id = int(current_jobs[machine_idx])
            assert job_id not in [-1, -2], "Job id should never be -1 or -2 at this point."


            task: Job = self.jobs[job_id]
            machine = self.machines[machine_idx]

            relevant_task: list[tuple[int, int, int]] = list()

            # Get the last job on the same machine
            latest_job_on_same_machine = schedule[machine_idx][-1]
            relevant_task.append(latest_job_on_same_machine)

            if len(task.dependencies) > 0:
                for dep in task.dependencies:
                    if dep_task := job_schedule.get(dep, None):
                        relevant_task.append(dep_task)
                    else:
                        raise ScheduleError(
                            f"Dependency {dep} not scheduled before {job_id}"
                        )

            # Get the start time of the task
            start_time = max([task[2] for task in relevant_task])

            task_duration: int = int(
                task.available_machines[machine_idx]
            )

            end_time = start_time + task_duration
            schedule[machine_idx].append((job_id, start_time, end_time))
            job_schedule[job_id] = (machine_idx, start_time, end_time)

            # Increase current index
            current_index[machine_idx] += 1
            current_jobs = job_orders[:,current_index].diagonal()
            jobs_assigned.add(job_id)

        return schedule

    def visualize_schedule(self, schedule: schedule_type, save_path: str | None = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap("tab10")
        for machine in schedule.keys():
            for job in schedule[machine]:
                if job[0] == -1:
                    continue
                ax.broken_barh([(job[1], job[2] - job[1])], (machine - 0.5, 1), facecolors=cmap(job[0] // self.number_of_tasks))
                ax.text(job[1] + (job[2] - job[1]) / 2, machine, str(job[0]), ha="center", va="center")
        plt.show()

def load_standard_data(file_path) -> SimpleJobShopProblem:
    with open(file_path, "r") as f:
        lines = f.readlines()
        n_jobs, n_machines = map(int, lines[0].split())
        jobs: list[Job] = list()
        for job_idx, str_jobs in enumerate(lines[1:]):
            job_string_split = str_jobs.split()
            for idx, (machine, job_time) in enumerate(
                zip(job_string_split[::2], job_string_split[1::2])
            ):
                dependencies = list()
                if idx > 0:
                    for i in range(len(jobs) - 1 + 1 - idx, len(jobs)):
                        dependencies.append(i)
                jobs.append(
                    Job(
                        dependencies=dependencies,
                        available_machines={int(machine): int(job_time)},
                        production_order_nr=str(job_idx),
                    )
                )
        machines = [
            Machine(
                name=str(i), machine_id=i, end_time=int(10e100), minutes_per_run=1.0
            )
            for i in range(n_machines)
        ]
        jssp = SimpleJobShopProblem(jobs=jobs, machines=machines, number_of_tasks=len(jobs) // n_jobs)
        return jssp

class OneStageACO(TwoStageACO):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # We assume that each job can only be assigned to one machine
        self.machine_assignment_set = {i: {idx for idx, j in enumerate(self.problem.jobs) if j.available_machines.get(i, None) is not None} for i in range(len(self.problem.machines))}
        self.machine_assignment = [list(i.available_machines.keys())[0] for i in self.problem.jobs]

    def draw_job_to_schedule(self, jobs_to_schedule: set[int], last: int, machine: int | None = None) -> int:
        jobs_to_schedule_list = list(jobs_to_schedule)
        if len(jobs_to_schedule_list) == 1:
            return jobs_to_schedule_list[0]

        probabilites = np.zeros(len(jobs_to_schedule_list))
        denominator = 0.
        for idx, job in enumerate(jobs_to_schedule_list):
            tau_r_s = self.pheromones_stage_two[last, job, 0]
            eta_r_s = 1. / self.problem.jobs[job].available_machines[self.machine_assignment[job]]
            probabilites[idx] = tau_r_s * eta_r_s
            denominator += probabilites[idx]

        # The pseudo-random number is used to determine if we should exploit or explore.
        if np.random.rand() <= self.q_zero:
            return jobs_to_schedule_list[np.argmax(probabilites)]
        # Avoid division by zero when we have very small numbers
        if denominator <= 1e-6:
            return select_random_item(jobs_to_schedule_list)
        probabilites = probabilites / denominator
        return select_random_item(jobs_to_schedule_list, probabilities=probabilites)
        

    def run_ant(self) -> tuple[np.ndarray, dict[int, set[int]]]:
            """Run the ant and return the job order and machine assignment.

            Returns:
                tuple[np.ndarray, dict[int, set[int]]]: job order and machine assignment.
            """
            machine_assignment = self.machine_assignment_set
            
            all_jobs_assigned = set()

            job_order = list()

            for _ in range(len(self.problem.jobs)):
                jobs_to_schedule = {
                    idx for idx, job in enumerate(self.problem.jobs) if set(job.dependencies).issubset(all_jobs_assigned)
                }.difference(all_jobs_assigned)
                if len(jobs_to_schedule) == 0:
                    raise ScheduleError("No job can be scheduled, even though jobs are left")
                job_idx = self.draw_job_to_schedule(
                    jobs_to_schedule=jobs_to_schedule,
                    last=job_order[-1] if len(job_order) > 0 else -1,
                )
                job_order.append(job_idx)
                all_jobs_assigned.add(job_idx)
            

            schedules = np.ones((len(self.problem.machines), len(self.problem.jobs)), dtype=int) * - 2
            # First column is -1
            schedules[:,0] = -1
            # Move from the linear schedule to the parallel schedule
            # The parallel schedule is a matrix where each row is a machine and each column is a job
            for job in job_order:
                machine = self.machine_assignment[job]
                idx = np.where(schedules[machine] == -2)[0][0]
                schedules[machine, idx] = job
            
            return schedules, machine_assignment

if __name__ == "__main__":
    problem = load_standard_data(r"B:\Documents\Skola\UvA\Y3P6\dev\src\examples\dmu60.txt")
    aco = OneStageACO(
        problem=problem,
        seed=34553,
        n_iter=1000,
        n_ants=36,
        tau_zero=1.0/4300.,
        q_zero=0.0,
        objective_function=ObjectiveFunction.MAKESPAN,
        verbose=True,
        with_local_search=False,
    )
    aco.run()
    best_schedule = aco.problem.make_schedule_from_parallel(aco.best_solution[1])
    aco.problem.visualize_schedule(best_schedule)


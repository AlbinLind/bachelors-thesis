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

from numpy import ndarray
import numpy as np
from src.production_orders import Data
from src.schedule_generator.ant_colony_optimisation import TwoStageACO
from src.schedule_generator.main import Job, Machine, JobShopProblem, ObjectiveFunction, ScheduleError, schedule_type

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
            
            print(f"{current_jobs=}, {current_index=}, {available_jobs=}, {self.number_of_tasks=}")
            # Get the job index with the lowest index
            lowest_indecies = np.where(available_jobs == available_jobs.min())
            lowest_index = lowest_indecies[0][0]
            if len(lowest_indecies[0]) > 1:
                lowest_index = np.array(current_index)[lowest_indecies].argmin()

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

        print(schedule)
        return schedule

def load_standard_data(file_path):
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
        print(len(jobs))
        jssp = SimpleJobShopProblem(jobs=jobs, machines=machines, number_of_tasks=len(jobs) // n_jobs)
        return jssp


if __name__ == "__main__":
    problem = load_standard_data(r"B:\Documents\Skola\UvA\Y3P6\dev\src\examples\mini.txt")
    aco = TwoStageACO(
        problem=problem,
        n_iter=1,
        n_ants=1,
        objective_function=ObjectiveFunction.MAKESPAN,
        verbose=True,
        with_local_search=False,
    )
    aco.run()

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

from src.schedule_generator.main import Job, Machine


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
                    dependencies.append(len(jobs) - 1)
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


if __name__ == "__main__":
    load_standard_data("examples/ft06.txt")

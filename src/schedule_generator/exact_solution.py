"""Mathematical model for the exact solution of the scheduling problem."""

import time
import typing
import pyomo.environ as pyo
from src.schedule_generator.main import JobShopProblem, schedule_type
from src.production_orders import parse_data
from pyomo.util.infeasible import log_infeasible_constraints
import logging

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logger.addHandler(ch)

H = 10e3


def generate_model(
    jssp: JobShopProblem, tardiness_objective: bool = True
) -> pyo.ConcreteModel:
    """Generates a pyomo model for the job shop scheduling problem.

    The model can either minimize the sum of tardiness or the makespan.
    Args:
        jssp (JobShopProblem): the job shop scheduling problem.
        tardiness_objective (bool, optional): True if objective function is sum tardiness, otherwise it is makespan. Defaults to True.

    Returns:
        pyo.ConcreteModel: unsolved mathematical pyomo model.
    """
    model = pyo.ConcreteModel()
    model.jobs = pyo.Set(initialize=range(len(jssp.jobs)))
    model.machines = pyo.Set(initialize=range(len(jssp.machines)))

    model.t = pyo.Var(model.jobs, domain=pyo.NonNegativeReals, initialize=360)
    model.alpha = pyo.Var(model.jobs, model.machines, domain=pyo.Binary, initialize=0)
    model.beta = pyo.Var(model.jobs, model.jobs, domain=pyo.Binary, initialize=0)
    model.sigma = pyo.Var(model.jobs, domain=pyo.Integers, initialize=0)
    model.w = pyo.Var(
        model.jobs, bounds=(0, 1), domain=pyo.NonNegativeReals, initialize=0
    )
    model.z = pyo.Var(
        model.jobs, bounds=(0, 1), domain=pyo.NonNegativeReals, initialize=0
    )
    model.mu = pyo.Var(model.jobs, bounds=(-2, 10), domain=pyo.Integers, initialize=-1)
    model.epsilon = pyo.Var(model.jobs, domain=pyo.NonNegativeReals, initialize=0)

    if tardiness_objective:
        model.tardiness = pyo.Var(model.jobs, domain=pyo.NonNegativeReals, initialize=0)

        model.objective = pyo.Objective(
            expr=pyo.quicksum(model.tardiness[j] for j in model.jobs),
            sense=pyo.minimize,
        )

        # Calculate tardiness for each job
        def calculate_tardiness(m, j):
            return (
                m.tardiness[j]
                >= m.t[j]
                + m.epsilon[j]
                + pyo.quicksum(
                    m.alpha[j, machine]
                    * jssp.jobs[j].available_machines.get(machine, 0)
                    for machine in m.machines
                    if machine in jssp.jobs[j].available_machines
                )
                - (jssp.jobs[j].days_till_delivery + 1) * 24 * 60
            )

        model.calculate_tardiness = pyo.Constraint(model.jobs, rule=calculate_tardiness)
    else:
        # Calculate makespan
        model.c_max = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)

        def calculate_makespan(m, j):
            return m.c_max >= (
                m.t[j]
                + pyo.quicksum(
                    m.alpha[j, machine]
                    * jssp.jobs[j].available_machines.get(machine, 0)
                    for machine in m.machines
                    if machine in jssp.jobs[j].available_machines
                )
                + m.epsilon[j]
            )

        model.calculate_makespan = pyo.Constraint(model.jobs, rule=calculate_makespan)
        model.objective = pyo.Objective(expr=model.c_max, sense=pyo.minimize)

    # Only one machine assigned to each job
    def one_machine_per_job(m, j):
        return (
            pyo.quicksum(
                m.alpha[j, machine]
                for machine in m.machines
                if machine in jssp.jobs[j].available_machines
            )
            == 1
        )

    model.one_machine_per_job = pyo.Constraint(model.jobs, rule=one_machine_per_job)

    # Keep precedence order of jobs
    def precedence_order(m, j1, j2):
        if len(jssp.jobs[j1].dependencies) > 0 and j2 in jssp.jobs[j1].dependencies:
            return m.t[j1] >= (
                m.t[j2]
                + m.epsilon[j2]
                + pyo.quicksum(
                    m.alpha[j2, machine]
                    * jssp.jobs[j2].available_machines.get(machine, 0)
                    for machine in m.machines
                    if machine in jssp.jobs[j2].available_machines
                )
            )
        else:
            return pyo.Constraint.Skip

    model.precedence_order = pyo.Constraint(
        model.jobs, model.jobs, rule=precedence_order
    )

    # Make sure that jobs does not overlap
    def no_overlapping_1(m, j1, j2, machine):
        if (
            j1 == j2
            or machine not in jssp.jobs[j1].available_machines
            or machine not in jssp.jobs[j2].available_machines
        ):
            return pyo.Constraint.Skip
        else:
            return (
                m.t[j1]
                >= m.t[j2]
                + jssp.jobs[j2].available_machines[machine]
                + jssp.setup_times[j2][j1]
                + m.epsilon[j2]
                - (2 - m.alpha[j1, machine] - m.alpha[j2, machine] + m.beta[j1, j2]) * H
            )

    model.no_overlapping_1 = pyo.Constraint(
        model.jobs, model.jobs, model.machines, rule=no_overlapping_1
    )

    def no_overlapping_2(m, j1, j2, machine):
        if (
            j1 == j2
            or machine not in jssp.jobs[j1].available_machines
            or machine not in jssp.jobs[j2].available_machines
        ):
            return pyo.Constraint.Skip
        else:
            return (
                m.t[j2]
                >= m.t[j1]
                + jssp.jobs[j1].available_machines[machine]
                + jssp.setup_times[j1][j2]
                + m.epsilon[j1]
                - (3 - m.alpha[j1, machine] - m.alpha[j2, machine] - m.beta[j1, j2]) * H
            )

    model.no_overlapping_2 = pyo.Constraint(
        model.jobs, model.jobs, model.machines, rule=no_overlapping_2
    )

    # Floor function for start time day
    def floor_function_start(m, j):
        return m.t[j] / (24 * 60) == m.sigma[j] + m.w[j]

    model.floor_function_start = pyo.Constraint(model.jobs, rule=floor_function_start)

    # Enforce start time is after start of machine
    def enforce_start_time(m, j):
        return m.t[j] >= 24 * 60 * m.sigma[j] + pyo.quicksum(
            m.alpha[j, machine] * (jssp.machines[machine].start_time)
            for machine in m.machines
            if machine in jssp.jobs[j].available_machines
        )

    model.enforce_start_time = pyo.Constraint(model.jobs, rule=enforce_start_time)

    # Enforce start time is before end of machine
    def enforce_end_time(m, j):
        return m.t[j] <= 24 * 60 * m.sigma[j] + pyo.quicksum(
            m.alpha[j, machine] * (jssp.machines[machine].end_time)
            for machine in m.machines
            if machine in jssp.jobs[j].available_machines
        )

    model.enforce_end_time = pyo.Constraint(model.jobs, rule=enforce_end_time)

    # Make sure the product is finished before the end of the day
    def end_of_day(m, j):
        return m.t[j] + pyo.quicksum(
            m.alpha[j, machine] * jssp.jobs[j].available_machines[machine]
            for machine in m.machines
            if machine in jssp.jobs[j].available_machines
        ) <= 24 * 60 * m.sigma[j] + pyo.quicksum(
            m.alpha[j, machine] * jssp.machines[machine].end_time
            for machine in m.machines
            if machine in jssp.jobs[j].available_machines
        )

    # model.end_of_day = pyo.Constraint(model.jobs, rule=end_of_day)

    # Floor function for end time day
    def floor_function_end(m, j):
        return (
            m.t[j]
            + pyo.quicksum(
                m.alpha[j, machine]
                * (
                    jssp.jobs[j].available_machines[machine]
                    - jssp.machines[machine].end_time
                )
                for machine in m.machines
                if machine in jssp.jobs[j].available_machines
            )
        ) / (24 * 60) == m.mu[j] + m.z[j]

    model.floor_function_end = pyo.Constraint(model.jobs, rule=floor_function_end)

    # Add extra time for downtime
    def add_extra_time(m, j):
        return m.epsilon[j] >= 24 * 60 * (m.mu[j] - m.sigma[j] + 1) + pyo.quicksum(
            m.alpha[j, machine]
            * (jssp.machines[machine].start_time - jssp.machines[machine].end_time)
            for machine in m.machines
            if machine in jssp.jobs[j].available_machines
        )

    model.add_extra_time = pyo.Constraint(model.jobs, rule=add_extra_time)

    return model


def solve_model(model: pyo.ConcreteModel, time_limit: int | None = None):
    """Solve a generated pyomo model. Please have CPLEX downloaded on your system.

    This function has to be modified if a solve different from CPLEX is to be used.
    """
    solver = pyo.SolverFactory("cplex")
    solver.options["timelimit"] = time_limit
    res = solver.solve(model, tee=True)
    if res.solver.status != pyo.SolverStatus.ok:
        print("Check solver not ok...")
        raise Exception("Solver not ok")
    if (
        res.solver.termination_condition != pyo.TerminationCondition.optimal
        and not time_limit
    ):
        print("Could not find optimal solution, probably infeasible...")
        log_infeasible_constraints(
            model, logger=logger, log_expression=True, log_variables=True
        )
        raise Exception("Infeasible solution")
    return model


@typing.no_type_check
def get_schedule(model: pyo.ConcreteModel, jssp: JobShopProblem) -> schedule_type:
    """Generate a schedule from the *solved* model.

    This function uses the start time from the solved model.
    """
    job_start_times: dict[int, float] = {j: model.t[j].value for j in model.jobs}
    schedule: schedule_type = {
        m: [(-1, 0, jssp.machines[m].start_time)] for m in model.machines
    }
    for j in sorted(job_start_times, key=job_start_times.get):
        found = False
        for m in model.machines:
            if model.alpha[j, m].value == 1 and not found:
                start_time = job_start_times[j]
                setup_time = 0
                if len(schedule[m]) > 1:
                    last_job_idx, _, _ = schedule[m][-1]
                    setup_time = jssp.setup_times[last_job_idx][j]
                end_time = start_time + jssp.jobs[j].available_machines[m]
                start_time, end_time = jssp._calculate_start_and_end_time(
                    machine_allow_preemption=True,
                    machine_start_time=jssp.machines[m].start_time,
                    machine_end_time=jssp.machines[m].end_time,
                    start_time=start_time,
                    task_duration=end_time - start_time,
                )
                schedule[m].append((j, start_time, end_time))
                found = True
            elif found and model.alpha[j, m].value == 1:
                print("Error: Multiple machines assigned to job")
    return schedule


def check_model_feasible(model: pyo.ConcreteModel) -> bool:
    for c in model.component_objects(pyo.Constraint, active=True):
        for i in c:
            if not c[i].body():
                print(f"Constraint {c.name} not satisfied for {i}")
                return False
    return True


if __name__ == "__main__":
    jssp = JobShopProblem.from_data(parse_data("examples/data_v1.xlsx"))
    print("Data parsed...")
    print("Generating model...")
    model = generate_model(jssp)
    print("Model generated...")
    print("Solving model...")
    start_time = time.time()
    solve_model(model, time_limit=3 * 60 * 60)
    end_time = time.time()
    print("Model solved...")
    sc = get_schedule(model, jssp)
    print(sc)
    print("Makespan: ", jssp.makespan(sc))
    print("Linear Makespan: ", model.objective())
    print("Time: ", end_time - start_time)
    jssp.visualize_schedule(sc)

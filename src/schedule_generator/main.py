from enum import Enum
from typing import Any, Self
import numpy as np
import pandas as pd
from pydantic import BaseModel
from src.production_orders import Data, Product, BillOfMaterial
import networkx as nx
import matplotlib.pyplot as plt

DAY_MINUTES = 24 * 60


class Job(BaseModel):
    """Contains the information about a job that needs to be done."""

    available_machines: dict[int, float]
    dependencies: list[int]
    production_order_nr: str
    station_settings: dict[str, Any] = dict()
    amount: int = 1
    days_till_delivery: int = 0
    used: float = 0.0


class Machine(BaseModel):
    """Contains information about a machine that can be used to process jobs."""

    name: str
    machine_id: int
    start_time: int = 0
    end_time: int
    allow_preemption: bool = False
    max_units_per_run: int = 1
    minutes_per_run: float


class ScheduleError(Exception): ...


schedule_type = dict[int, list[tuple[int, int, int]]]


class JobShopProblem:
    LOW_TARDINESS = None
    LOW_TOTAL_SETUP_TIME = None
    LOW_MAKESPAN = None

    def __init__(self, data: Data, jobs: list[Job], machines: list[Machine]) -> None:
        self.data: Data = data
        self.jobs: list[Job] = jobs
        self.machines: list[Machine] = machines
        self.setup_times: np.ndarray = np.zeros((len(jobs), len(jobs)))
        self.graph = self._build_graph()
        self.bottle_size_mapping: dict[str, float] = {
            product.setting_bottle_size: data.bill_of_materials[
                product.product_id
            ].component_quantity
            if product.setting_bottle_size
            else 1.0
            for product in data.products.values()
            if product.setting_bottle_size
        }

    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from([-1, -2] + [x for x in range(len(self.jobs))])
        edges = list()
        for job_idx, job in enumerate(self.jobs):
            if len(job.dependencies) == 0:
                edges.append((-1, job_idx))
                continue
            for dep in job.dependencies:
                edges.append((dep, job_idx))
        graph.add_edges_from(edges)
        for node, outdegree in graph.out_degree(graph.nodes()):
            if outdegree == 0 and node >= 0:
                graph.add_edge(node, -2)
        return graph

    def visualize_schedule(
        self,
        schedule: dict[int, list[tuple[int, int, int]]],
        save_path: str | None = None,
    ):
        """Generates a visualization of the schedule.

        Args:
            schedule (dict[int, list[tuple[int, int, int]]]): the schedule that should be visualized
            save_path (str | None, optional): path to save the visualisation, if none is give it will display it.
                Defaults to None.
        """
        fig, ax = plt.subplots(figsize=(18, 7))
        cmap = plt.get_cmap("tab10")
        flavour_mapping = {"cola": 0, "fanta": 1, "orange juice": 2, "apple juice": 3}
        for i, (machine, sch) in enumerate(schedule.items()):
            for idx, task in enumerate(sch):
                job_id, start_time, end_time = task
                if job_id == -1:
                    continue
                setup_time = 0
                if sch[idx - 1][0] != -1:
                    setup_time = self.setup_times[sch[idx - 1][0], job_id]
                flavour = self.jobs[job_id].station_settings["taste"]
                to_be_plotted = list()
                if self.machines[machine].name[0] == "M":
                    # Get hf flavour
                    to_be_plotted.append((start_time + setup_time, end_time, False))
                    if setup_time > 0:
                        to_be_plotted.append(
                            (start_time, start_time + setup_time, True)
                        )
                else:
                    # List with tuples of (start time, end time, is setup?)
                    # Check if we need to split into multiple parts because of preemption
                    if (
                        end_time - start_time - setup_time
                        > self.jobs[job_id].available_machines[machine]
                    ):
                        # How many days are we splitting it into?
                        no_days = (
                            end_time - start_time - setup_time
                        ) // DAY_MINUTES + 1
                        start_day = start_time // DAY_MINUTES
                        if no_days == 1:
                            if (start_time + setup_time) % DAY_MINUTES > self.machines[
                                machine
                            ].end_time:
                                to_be_plotted.append(
                                    (
                                        start_time,
                                        self.machines[machine].end_time
                                        + DAY_MINUTES * start_day,
                                        True,
                                    )
                                )
                                to_be_plotted.append(
                                    (
                                        self.machines[machine].start_time
                                        + DAY_MINUTES * (start_day + 1),
                                        self.machines[machine].start_time
                                        + DAY_MINUTES * (start_day + 1)
                                        + (DAY_MINUTES - start_time + setup_time)
                                        % DAY_MINUTES,
                                        True,
                                    )
                                )
                                to_be_plotted.append(
                                    (
                                        self.machines[machine].start_time
                                        + DAY_MINUTES * (start_day + 1),
                                        end_time,
                                        False,
                                    )
                                )
                            else:
                                to_be_plotted.append(
                                    (start_time, start_time + setup_time, True)
                                )
                                to_be_plotted.append(
                                    (
                                        start_time + setup_time,
                                        self.machines[machine].end_time
                                        + DAY_MINUTES * start_day,
                                        False,
                                    )
                                )
                                to_be_plotted.append(
                                    (
                                        self.machines[machine].start_time
                                        + DAY_MINUTES * (start_day + 1),
                                        end_time,
                                        False,
                                    )
                                )

                    else:
                        if setup_time > 0:
                            to_be_plotted.append(
                                (start_time, start_time + setup_time, True)
                            )
                        to_be_plotted.append((start_time + setup_time, end_time, False))
                for current_plot in to_be_plotted:
                    kwargs = {
                        # "hatch": "O",
                        "facecolor": cmap(flavour_mapping[flavour]),
                        "edgecolor": "black",
                        "label": flavour,
                    }
                    if current_plot[2]:
                        kwargs["hatch"] = "/"
                        kwargs["alpha"] = 0.5
                        kwargs["facecolor"] = "gray"
                        kwargs["label"] = "setup"
                    elif self.machines[machine].name[0] == "B":
                        color = "white"
                        if (
                            end_time
                            - (self.jobs[job_id].days_till_delivery + 1) * 24 * 60
                            > 0
                        ):
                            color = "red"
                        ax.text(
                            (current_plot[0] + current_plot[1]) / 2,
                            i,
                            self.jobs[job_id].production_order_nr,
                            va="center",
                            ha="center",
                            fontsize=21,
                            color=color,
                        )
                    ax.broken_barh(
                        [(current_plot[0], current_plot[1] - current_plot[0])],
                        (i - 0.25, 0.5),
                        **kwargs,
                    )
        max_time = max(
            [schedule[machine.machine_id][-1][2] for machine in self.machines]
        )
        for machine in self.machines:
            x_lines_start = np.arange(machine.start_time, max_time, 24 * 60)
            plt.vlines(
                x_lines_start,
                machine.machine_id - 0.4,
                machine.machine_id + 0.4,
                linestyles="dashed",
                color="green",
            )
            x_lines_end = np.arange(machine.end_time, max_time, 24 * 60)
            plt.vlines(
                x_lines_end,
                machine.machine_id - 0.4,
                machine.machine_id + 0.4,
                linestyles="dashed",
                color="red",
            )

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # Machine labels
        plt.yticks(
            ticks=[m.machine_id for m in self.machines],
            labels=[str(self.machines[m].name) for m in schedule.keys()],
        )
        plt.ylabel("Machine")

        plt.xlabel("Time (minutes)")

        # Add box with information about the schedule in the bottom right corner
        textstr = f"Total tardiness: {self.classical_tardiness(schedule)}\nBoolean tardiness: {self.boolean_tardiness(schedule)}\nTotal setup time: {self.total_setup_time(schedule)}\nMakespan: {self.makespan(schedule)}"
        props = dict(boxstyle="round", facecolor="gray", alpha=0.3)
        plt.text(
            0.99,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def old_visualize_schedule(
        self,
        schedule: dict[int, list[tuple[int, int, int]]],
        save_path: str | None = None,
    ):
        """Visualizes a schedule."""
        fig, ax = plt.subplots(figsize=(13, 7))
        cmap = plt.get_cmap("tab20")
        for i, (machine, sch) in enumerate(schedule.items()):
            for idx, task in enumerate(sch):
                job_id, start_time, end_time = task
                if job_id == -1:
                    continue
                setup_time = 0
                if idx > 1:
                    setup_time = self.setup_times[sch[idx - 1][0], job_id]
                # Check if we have a preemption job
                plot_times = [(start_time, end_time, setup_time)]
                if (
                    end_time - start_time - setup_time
                    > self.jobs[job_id].available_machines[machine]
                ):
                    plot_times = [
                        (
                            start_time,
                            self.machines[machine].end_time
                            + 24 * 60 * (start_time // (24 * 60)),
                            setup_time,
                        ),
                        (
                            self.machines[machine].start_time
                            + 24 * 60 * (end_time // (24 * 60)),
                            end_time,
                            0,
                        ),
                    ]
                for start_time, end_time, setup_time in plot_times:
                    ax.plot(
                        [start_time + setup_time, end_time],
                        [i + 1, i + 1],
                        linewidth=50,
                        label=self.jobs[job_id].production_order_nr,
                        solid_capstyle="butt",
                        color=cmap(
                            int(self.jobs[job_id].production_order_nr.removeprefix("P"))
                        ),
                    )
                    ax.plot(
                        [start_time, start_time + setup_time],
                        [i + 1, i + 1],
                        linewidth=50,
                        solid_capstyle="butt",
                        color=cmap(
                            int(self.jobs[job_id].production_order_nr.removeprefix("P"))
                        ),
                        alpha=0.5,
                    )
                    color = "black"
                    if (
                        end_time - (self.jobs[job_id].days_till_delivery + 1) * 24 * 60
                        > 0
                    ):
                        color = "red"

                    ax.text(
                        (start_time + end_time) / 2,
                        i + 1,
                        self.jobs[job_id].production_order_nr,  # + f" ({job_id})",
                        va="center",
                        ha="right",
                        fontsize=11,
                        color=color,
                    )
        flat_schedule = list()
        for val in schedule.values():
            flat_schedule.extend(val)
        max_time = max([t[2] for t in flat_schedule])

        day_markers = np.arange(0, max_time, 24 * 60)
        day_labels = [f"{d//24//60}" for d in day_markers]

        plt.xticks(ticks=np.concatenate([day_markers]), labels=day_labels)
        plt.yticks(
            ticks=np.arange(1, len(schedule) + 1),
            labels=[str(self.machines[m].name) for m in schedule.keys()],
        )
        plt.xlabel("Days")
        plt.ylabel("Machine")
        plt.tight_layout()

        for machine in self.machines:
            x_lines_start = np.arange(machine.start_time, max_time, 24 * 60)
            plt.vlines(
                x_lines_start,
                machine.machine_id + 0.5,
                machine.machine_id + 1.5,
                linestyles="dashed",
                color="green",
            )
            x_lines_end = np.arange(machine.end_time, max_time, 24 * 60)
            plt.vlines(
                x_lines_end,
                machine.machine_id + 0.5,
                machine.machine_id + 1.5,
                linestyles="dashed",
                color="red",
            )

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @classmethod
    def from_data(cls, data: Data) -> Self:
        """Generate a JobShopProblem from the data provided according to the excel sheet.

        Args:
            data (Data): The data that is parsed from the excel sheet.

        Returns:
            Self: The JobShopProblem object.
        """
        sub_jobs: list[Job] = list()
        machines: list[Machine] = [
            Machine(
                name=m.name,
                machine_id=idx,
                start_time=m.starts_at.hour * 60 + m.starts_at.minute,
                end_time=m.stops_at.hour * 60 + m.stops_at.minute,
                allow_preemption=(m.name.lower().startswith("bottling")),
                max_units_per_run=m.max_units_per_run,
                minutes_per_run=m.minutes_per_run,
            )
            for idx, m in enumerate(data.workstations)
        ]
        for order in data.production_orders:
            # We collect all the products we need to produce
            # 1. We get the final product from the products table
            # 2. We look in the bill of materials to see if that product has any sub products that are needed
            # 3. If it has, we add the sub product from the products table and repeat step 2.
            products = list()
            curent_product = order.product_id
            while curent_product is not None:
                product = data.products[curent_product]
                bill_of_materials = data.bill_of_materials.get(curent_product, None)
                products.append((product, bill_of_materials))
                if bill_of_materials:
                    curent_product = bill_of_materials.component_id
                else:
                    curent_product = None
            # We reverse the products list, because the products have to be produced in sequential order
            products: list[tuple[Product, BillOfMaterial | None]] = products[::-1]
            # We split the job into batches of the same product
            product_info = list()
            # NOTE: amount is different from the amount in the job, since we calcualte the
            # amount based on what unit the machine can process
            for idx, (prod, _) in enumerate(products):
                prod_info = dict()
                if idx + 1 < len(products):
                    bom = products[idx + 1][1]
                else:
                    bom = None
                amount = order.amount * bom.component_quantity if bom else order.amount
                prod_info["amount"] = amount
                # HACK: we are shortening the name by 1 since it says bottle and not bottling, which the machine name is
                prod_info["machines"] = [
                    m.machine_id
                    for m in machines
                    if m.name.lower().startswith(prod.workstation_type[:-1])
                ]
                # HACK: the batch sizes are not calculated per machine, so if two machines
                # that process the same job has different ones, we will take the one that has
                # the lowest capacity to calculate the batch size
                min_max_units_per_run = min(
                    [machines[m].max_units_per_run for m in prod_info["machines"]]
                )
                batches = np.ceil(amount / min_max_units_per_run)
                prod_info["batches"] = int(batches)
                prod_info["batches_amount"] = min_max_units_per_run
                remainder = int(amount % min_max_units_per_run)
                prod_info["batches_remainder"] = (
                    remainder if remainder > 0 else min_max_units_per_run
                )
                product_info.append(prod_info)
            # We calculate the correct batch size, or in other words the job that has
            # the smallest batch amount size, since that will be the bottle neck.
            # However, we ignore if the jobs can be run independently, since we will just make that into
            # one batch
            batch_info = min(
                product_info,
                key=lambda x: x["batches_amount"] if x["batches_amount"] > 1 else 10e4,
            )
            for i in range(batch_info["batches"]):
                for idx, prod in enumerate(product_info):
                    dependencies = list()
                    if idx > 0:
                        dependencies.append(len(sub_jobs) - 1)

                    amount = prod["amount"] // batch_info["batches"]

                    if i == batch_info["batches"] - 1 and batch_info["batches"] > 1:
                        amount = prod["amount"] % ((batch_info["batches"] - 1) * amount)
                        if amount == 0:
                            amount = prod["amount"] // batch_info["batches"]
                    amount = np.ceil(amount)
                    sub_jobs.append(
                        Job(
                            available_machines={
                                m: data.workstations[m].minutes_per_run * amount
                                if data.workstations[m].max_units_per_run == 1
                                else data.workstations[m].minutes_per_run
                                for m in prod["machines"]
                            },
                            dependencies=dependencies,
                            production_order_nr=order.production_order_nr,
                            station_settings={
                                "taste": products[idx][0].setting_taste,
                                "bottle_size": products[idx][0].setting_bottle_size,  # type: ignore
                            },
                            amount=amount,
                            days_till_delivery=order.days_till_delivery,
                        )
                    )
        jssp = cls(data=data, jobs=sub_jobs, machines=machines)

        # Set setup-times
        for j1_idx, j1 in enumerate(jssp.jobs):
            for j2_idx, j2 in enumerate(jssp.jobs):
                if j1 == j2 or j1.production_order_nr == j2.production_order_nr:
                    continue
                if j1.station_settings["taste"] != j2.station_settings["taste"]:
                    jssp.setup_times[j1_idx, j2_idx] += jssp.data.workstations[
                        list(j1.available_machines.keys())[0]
                    ].minutes_changeover_time_taste

                if (
                    j1.station_settings["bottle_size"]
                    != j2.station_settings["bottle_size"]
                ):
                    jssp.setup_times[j1_idx, j2_idx] += jssp.data.workstations[
                        list(j1.available_machines.keys())[0]
                    ].minutes_changeover_time_bottle_size

        return jssp

    def make_schedule(
        self, job_order: list[int], machine_assignment: list[int]
    ) -> schedule_type:
        """Create a schedule based on a given job order and machine assignment.
        It is advised to not use this function, but rather use the `make_schedule_from_parallel`
        or `make_schedule_from_parallel_with_stock`.

        Note that the job_order is relative, and machine_assignment is absolute. That means that
        the machine_assignment have at index i the machine assignment for job i. While the job_order
        at index i is the job that should be done at position i in relation to the other jobs in the list.

        Args:
            job_order (list[int]): the order of the jobs that should be done
            machine_assignment (list[int]): what machine job i should be done on

        Raises:
            ScheduleError: raised if the job order or machine assignment is incorrect

        Returns:
            dict[int, list[tuple[int, int, int]]]: the schedule for each machine with the job id, start time and end time
                relative to midnight of day 0.
        """
        # Contains the schedule for each machine
        schedule: dict[int, list[tuple[int, int, int]]] = {
            m.machine_id: [(-1, 0, m.start_time)] for m in self.machines
        }

        # Contains the machine, start time and end time for each job
        job_schedule: dict[int, tuple[int, int, int]] = dict()

        for task_idx in job_order:
            task: Job = self.jobs[task_idx]
            machine_idx = machine_assignment[task_idx]
            if machine_idx not in task.available_machines:
                raise ScheduleError(
                    f"Machine {machine_idx} not available for task {task_idx}"
                )
            machine = self.machines[machine_idx]

            relevant_task: list[tuple[int, int, int]] = list()

            # Get the last job on the same machine
            latest_job_on_same_machine = schedule[machine_idx][-1]
            relevant_task.append(latest_job_on_same_machine)

            # Check for dependencies
            if len(task.dependencies) > 0:
                for dep in task.dependencies:
                    if dep_task := job_schedule.get(dep, None):
                        relevant_task.append(dep_task)
                    else:
                        raise ScheduleError(
                            f"Dependency {dep} not scheduled before {task_idx}"
                        )

            # Get the start time of the task
            start_time = max([task[2] for task in relevant_task])

            # Only add setup time if we have a previous task
            setup_time: int = 0
            if len(schedule[machine_idx]) > 1:
                setup_time: int = self.setup_times[
                    schedule[machine_idx][-1][0], task_idx
                ]
            task_duration: int = int(task.available_machines[machine_idx] + setup_time)
            # If task is schedule before the machine starts, we move it to the start time
            if start_time % DAY_MINUTES < machine.start_time:
                start_time = (
                    machine.start_time + (start_time // DAY_MINUTES) * DAY_MINUTES
                )

            # If the task ends after the machine stops, we move it to the next day, unless we allow preemption.
            # If we allow preemption we will just continue with the work the next day
            if start_time % DAY_MINUTES + task_duration > machine.end_time:
                # If we allow for preemption we will just add to the duration the time inbetween start and end time
                if machine.allow_preemption:
                    task_duration += DAY_MINUTES - machine.end_time + machine.start_time
                else:
                    start_time = (
                        machine.start_time
                        + (start_time // DAY_MINUTES + 1) * DAY_MINUTES
                    )

            end_time = start_time + task_duration
            schedule[machine_idx].append((task_idx, start_time, end_time))
            job_schedule[task_idx] = (machine_idx, start_time, end_time)

        return schedule

    def make_schedule_from_parallel(
        self, job_orders: list[list[int]] | np.ndarray
    ) -> schedule_type:
        """Generates a schedule based on a parallel job order. This function takes
        into account the internal precedence constraints of the jobs. That means no
        job will be scheduled before its dependencies.

        Args:
            job_orders (list[list[int]] | np.ndarray): job orders for each machine

        Raises:
            ScheduleError: raised if the job order is incorrect

        Returns:
            schedule_type: generated schedule
        """
        schedule: dict[int, list[tuple[int, int, int]]] = {
            m.machine_id: [(-1, 0, m.start_time)] for m in self.machines
        }
        job_schedule: dict[int, tuple[int, int, int]] = dict()
        for machine in self.machines:
            machine_idx = machine.machine_id
            for task_idx in job_orders[machine_idx]:
                if task_idx == -1:
                    continue
                if task_idx == -2:
                    break
                task: Job = self.jobs[task_idx]
                if machine_idx not in task.available_machines:
                    raise ScheduleError(
                        f"Machine {machine_idx} not available for task {task_idx}"
                    )
                machine = self.machines[machine_idx]

                relevant_task: list[tuple[int, int, int]] = list()

                # Get the last job on the same machine
                latest_job_on_same_machine = schedule[machine_idx][-1]
                relevant_task.append(latest_job_on_same_machine)

                # Check for dependencies
                if len(task.dependencies) > 0:
                    for dep in task.dependencies:
                        if dep_task := job_schedule.get(dep, None):
                            relevant_task.append(dep_task)
                        else:
                            raise ScheduleError(
                                f"Dependency {dep} not scheduled before {task_idx}"
                            )

                # Get the start time of the task
                start_time = max([task[2] for task in relevant_task])

                # Only add setup time if we have a previous task
                setup_time = 0
                if len(schedule[machine_idx]) > 1:
                    setup_time: int = self.setup_times[
                        schedule[machine_idx][-1][0], task_idx
                    ]
                task_duration: int = int(
                    task.available_machines[machine_idx] + setup_time
                )
                # If task is schedule before the machine starts, we move it to the start time
                if start_time % DAY_MINUTES < machine.start_time:
                    start_time = (
                        machine.start_time + (start_time // DAY_MINUTES) * DAY_MINUTES
                    )

                # If the task ends after the machine stops, we move it to the next day, unless we allow preemption.
                # If we allow preemption we will just continue with the work the next day
                if start_time % DAY_MINUTES + task_duration > machine.end_time:
                    # If we allow for preemption we will just add to the duration the time inbetween start and end time
                    if machine.allow_preemption:
                        task_duration += (
                            DAY_MINUTES - machine.end_time + machine.start_time
                        )
                    else:
                        start_time = (
                            machine.start_time
                            + (start_time // DAY_MINUTES + 1) * DAY_MINUTES
                        )

                end_time = start_time + task_duration
                schedule[machine_idx].append((task_idx, start_time, end_time))
                job_schedule[task_idx] = (machine_idx, start_time, end_time)

        return schedule

    def _calculate_start_and_end_time(
        self,
        machine_allow_preemption: bool,
        machine_start_time: int,
        machine_end_time: int,
        start_time: float,
        task_duration: float,
    ) -> tuple[int, int]:
        """Calculate the start and end time of a task based on the machine settings.

        This function takes into account if the machine allows for preemption, and if the task can be done in one day etc.

        Args:
            machine_allow_preemption (bool): if the machine allows for preemption
            machine_start_time (int): start time of operating hours
            machine_end_time (int): end time of operating hours
            start_time (float): job start time
            task_duration (float): total duration of the task

        Returns:
            tuple[int, int]: (start_time, end_time)
        """
        start_time_remainder = start_time % DAY_MINUTES
        if start_time_remainder < machine_start_time:
            start_time = machine_start_time + (start_time // DAY_MINUTES) * DAY_MINUTES
            start_time_remainder = start_time % DAY_MINUTES

        if start_time_remainder + task_duration > machine_end_time:
            if machine_allow_preemption:
                task_duration += DAY_MINUTES - machine_end_time + machine_start_time
                while (
                    start_time_remainder + task_duration
                ) % DAY_MINUTES > machine_end_time:
                    task_duration += DAY_MINUTES - machine_end_time + machine_start_time

            else:
                start_time = (
                    machine_start_time + (start_time // DAY_MINUTES + 1) * DAY_MINUTES
                )
        return int(start_time), int(start_time + task_duration)

    def make_schedule_from_parallel_with_stock(
        self, job_orders: np.ndarray | list[list[int]]
    ) -> schedule_type:
        """Generate a schedule based on a parallel job order with stock.

        Args:
            job_orders (np.ndarray | list[list[int]]): job orders for each machine

        Raises:
            ScheduleError: if the stock is not enough to produce the product

        Returns:
            schedule_type: generated schedule
        """
        if isinstance(job_orders, list):
            max_length = max([len(j) for j in job_orders])
            job_orders = np.array(
                [j + [-2] * (max_length - len(j)) for j in job_orders]
            )
        schedule: dict[int, list[tuple[int, int, int]]] = {
            m.machine_id: [(-1, 0, m.start_time)] for m in self.machines
        }
        stock: dict[int, float] = {
            p: 0.0 for p in set([j.station_settings["taste"] for j in self.jobs])
        }
        available_jobs: dict[int, int] = {m: 0 for m in range(len(self.machines))}
        # A list with release times, hf_product, amount of stock that soon will be produced
        awaiting_release: list[tuple[int, int, float]] = list()
        for _ in range(len(self.jobs)):
            # Contains the machine_idx, job_idx, start_time of the job
            jobs_to_choose_from: list[tuple[int, int, int]] = list()
            # machine_that_finishes_soonest = sorted(schedule, key=lambda x: schedule[x][-1][2])
            for machine_idx in available_jobs:
                # Get the next job that should be done on the machine
                next_job_idx = available_jobs[machine_idx]
                if next_job_idx >= len(job_orders[machine_idx]):
                    continue
                task_idx: int = job_orders[machine_idx, next_job_idx]
                if task_idx == -2:
                    continue
                if task_idx == -1:
                    available_jobs[machine_idx] += 1
                    next_job_idx = available_jobs[machine_idx]
                    task_idx = job_orders[machine_idx, next_job_idx]
                    if task_idx == -2:
                        continue

                task: Job = self.jobs[task_idx]
                machine: Machine = self.machines[machine_idx]
                latest_job_on_same_machine = schedule[machine_idx][-1]
                start_time = latest_job_on_same_machine[2]

                if machine.name[0] == "M":
                    jobs_to_choose_from.append((machine_idx, task_idx, start_time))
                    continue
                elif machine.name[0] == "B":
                    # Check if we have enough stock to produce the product
                    amount_needed = (
                        task.amount
                        * self.bottle_size_mapping[task.station_settings["bottle_size"]]
                    )
                    stock_available = stock[task.station_settings["taste"]]
                    if stock_available >= amount_needed:
                        jobs_to_choose_from.append((machine_idx, task_idx, start_time))
                        continue
                    else:
                        # Check if we have any awaiting release
                        if len(awaiting_release) > 0:
                            awaiting_release.sort(key=lambda x: x[0])
                            for release_time, hf_product, amount in awaiting_release:
                                if hf_product != task.station_settings["taste"]:
                                    continue
                                if release_time <= start_time:
                                    stock_available += amount
                                    if stock_available >= amount_needed:
                                        jobs_to_choose_from.append(
                                            (machine_idx, task_idx, start_time)
                                        )
                                        break
                                else:
                                    stock_available += amount
                                    if stock_available >= amount_needed:
                                        jobs_to_choose_from.append(
                                            (machine_idx, task_idx, release_time)
                                        )
                                        break

            # Take the job that can start the soonest
            chosen_job = min(jobs_to_choose_from, key=lambda x: x[2])
            machine: Machine = self.machines[chosen_job[0]]
            task: Job = self.jobs[chosen_job[1]]

            # Only add setup time if we have a previous task
            setup_time: int = 0
            if len(schedule[machine.machine_id]) > 1:
                setup_time: int = self.setup_times[
                    schedule[machine.machine_id][-1][0], chosen_job[1]
                ]

            # Case for mixing line
            if machine.name[0] == "M":
                start_time = chosen_job[2]
                task_duration = task.available_machines[machine.machine_id] + setup_time
                end_time = chosen_job[2] + task_duration
                start_time, end_time = self._calculate_start_and_end_time(
                    machine.allow_preemption,
                    machine.start_time,
                    machine.end_time,
                    start_time,
                    task_duration,
                )
                schedule[machine.machine_id].append(
                    (chosen_job[1], start_time, end_time)
                )
                awaiting_release.append(
                    (
                        end_time,
                        task.station_settings["taste"],
                        machine.max_units_per_run,
                    )
                )

            # Case for bottling line
            elif machine.name[0] == "B":
                amount_needed = (
                    task.amount
                    * self.bottle_size_mapping[task.station_settings["bottle_size"]]
                )
                to_be_removed = list()
                for release_time, hf_product, amount in awaiting_release:
                    if (
                        hf_product == task.station_settings["taste"]
                        and release_time <= chosen_job[2]
                    ):
                        stock[task.station_settings["taste"]] += amount
                        to_be_removed.append((release_time, hf_product, amount))
                for r in to_be_removed:
                    awaiting_release.remove(r)
                stock_available = stock[task.station_settings["taste"]]
                if stock_available < amount_needed:
                    raise ScheduleError(
                        f"Stock is not enough to produce {task.production_order_nr}, this shouldn't happen..."
                    )
                start_time = chosen_job[2]
                task_duration = task.available_machines[machine.machine_id] + setup_time
                end_time = chosen_job[2] + task_duration
                start_time, end_time = self._calculate_start_and_end_time(
                    machine.allow_preemption,
                    machine.start_time,
                    machine.end_time,
                    start_time,
                    task_duration,
                )
                schedule[machine.machine_id].append(
                    (chosen_job[1], start_time, end_time)
                )
                stock[task.station_settings["taste"]] -= amount_needed
            available_jobs[machine.machine_id] += 1

        return schedule

    def makespan(self, schedule: schedule_type) -> int:
        """Calculate the makespan of the schedule.

        Args:
            schedule (schedule): the schedule that should be evaluated

        Returns:
            int: the makespan of the schedule
        """
        return max([machine_schedule[-1][2] for machine_schedule in schedule.values()])

    def tardiness(self, schedule: schedule_type) -> int:
        """Calculate the tardiness of the schedule. The tardiness is the number of sub jobs that are late.

        As soon as just 1 sub job is late the whole order is late, and should be penalized.

        Args:
            schedule (schedule): the schedule that should be evaluated

        Returns:
            int: the tardiness of the schedule
        """
        production_order_lateness = {
            order.production_order_nr: list() for order in self.data.production_orders
        }
        for machine in schedule.values():
            for task in machine:
                production_order_lateness[
                    self.jobs[task[0]].production_order_nr
                ].append(
                    max(
                        task[2]
                        - (self.jobs[task[0]].days_till_delivery + 1) * DAY_MINUTES,
                        0,
                    )
                )

        tardiness = 0
        for lateness in production_order_lateness.values():
            if any([l > 0 for l in lateness]):  # noqa: E741
                tardiness += np.max(lateness) * len(lateness)
        return int(tardiness)

    def classical_tardiness(self, schedule: schedule_type) -> float:
        """Calculates the classical tardiness of the schedule. I.e. the sum in minutes
        since the job was supposed to be done.

        Args:
            schedule (schedule_type): _description_

        Returns:
            float: _description_
        """
        return sum(
            [
                max(
                    task[2] - (self.jobs[task[0]].days_till_delivery + 1) * DAY_MINUTES,
                    0,
                )
                for machine in schedule.values()
                for task in machine
            ]
        )

    def total_setup_time(self, schedule: schedule_type) -> int:
        """Calculate the total setup time of the schedule.

        Args:
            schedule (schedule): the schedule that should be evaluated

        Returns:
            int: the total setup time of the schedule
        """
        setup_time = 0
        for machine in schedule.values():
            for idx, task in enumerate(machine):
                if idx > 0 and machine[idx - 1][0] != -1 and task[0] != -1:
                    setup_time += self.setup_times[machine[idx - 1][0], task[0]]
        return setup_time

    def custom_objective(self, schedule: schedule_type) -> float:
        """Calculate the custom objective function of the schedule.

        Args:
            schedule (schedule_type): schedule to be evaluated

        Returns:
            float: result of the evaluation
        """
        tardiness = self.boolean_tardiness(schedule)
        total_setup_time = self.total_setup_time(schedule)
        makespan = self.makespan(schedule)
        if self.LOW_TARDINESS is None:
            # Normal
            # self.LOW_TARDINESS = 8.0

            # Small
            # self.LOW_TARDINESS = 0.1

            # Unkown means we will take 60% of the total tardiness
            self.LOW_TARDINESS = 0.6 * tardiness
        if self.LOW_TOTAL_SETUP_TIME is None:
            # Normal
            # self.LOW_TOTAL_SETUP_TIME = 95.0

            # Small
            # self.LOW_TOTAL_SETUP_TIME = 35.0
            self.LOW_TOTAL_SETUP_TIME = 0.6 * total_setup_time
        if self.LOW_MAKESPAN is None:
            # Normal
            # self.LOW_MAKESPAN = 3502.0

            # Small
            # self.LOW_MAKESPAN = 2279.0
            self.LOW_MAKESPAN = 0.6 * makespan

        return (
            (tardiness - self.LOW_TARDINESS) / self.LOW_TARDINESS
            + (total_setup_time - self.LOW_TOTAL_SETUP_TIME) / self.LOW_TOTAL_SETUP_TIME
            + (makespan - self.LOW_MAKESPAN) / self.LOW_MAKESPAN
        )

    def boolean_tardiness(self, schedule: schedule_type) -> int:
        """Calculates the total number of days that a production order is late (includes all
        orders even if some are done on time).

        Args:
            schedule (schedule_type): schedule to be evaluated

        Returns:
            int: result of the evaluation
        """
        production_order_lateness = {
            order.production_order_nr: [] for order in self.data.production_orders
        }
        for machine in schedule.values():
            for task in machine:
                production_order_lateness[
                    self.jobs[task[0]].production_order_nr
                ].append(
                    max(
                        task[2]
                        - (self.jobs[task[0]].days_till_delivery + 1) * DAY_MINUTES,
                        0,
                    )
                )

        tardiness = 0
        for lateness in production_order_lateness.values():
            bool_lateness = [l > 0 for l in lateness]  # noqa: E741
            if any(bool_lateness):
                tardiness += (max(lateness) // DAY_MINUTES + 1) * len(lateness)
        return tardiness

    def generate_output(self, schedule: schedule_type | np.ndarray) -> pd.DataFrame:
        """Generat the output of the schedule in a standard format.

        Args:
            schedule (schedule_type | np.ndarray): either a generated schedule or a matrix with the job order

        Returns:
            pd.DataFrame: contains the workstation, product_id and amount of each job
        """
        info = {"workstation": [], "product_id": [], "amount": []}
        if isinstance(schedule, np.ndarray):
            schedule = self.make_schedule_from_parallel_with_stock(schedule)
        for machine, sch in schedule.items():
            for task in sch:
                if task[0] == -1:
                    continue
                info["workstation"].append(self.machines[machine].name)
                production_order_nr = self.jobs[task[0]].production_order_nr
                product_id = self.data.production_orders_df[
                    self.data.production_orders_df["production_order_nr"]
                    == production_order_nr
                ]["product_id"].values[0]
                # If we have a mixing line we need the hf product id
                if self.machines[machine].name[0] == "M":
                    product_id = self.data.bill_of_materials[product_id].component_id

                info["product_id"].append(product_id)
                info["amount"].append(self.jobs[task[0]].amount)

        return pd.DataFrame(info)


class ObjectiveFunction(Enum):
    CUSTOM_OBJECTIVE = 0
    MAKESPAN = 1
    TARDINESS = 2
    TOTAL_SETUP_TIME = 3
    BOOLEAN_TARDINESS = 4
    CLASSICAL_TARDINESS = 5


if __name__ == "__main__":
    from src.production_orders import parse_data

    data = parse_data(
        r"B:\Documents\Skola\UvA\Y3P6\git_folder\src\examples\data_v1.xlsx"
    )
    jssp = JobShopProblem.from_data(data)
    # sc = jssp.make_schedule_from_parallel_with_stock(np.array([[-1,0,10,4],[-1,5,-2,-2],[-1,1,-2,-2]]))
    # print(sc)
    # jssp.visualize_schedule(sc)

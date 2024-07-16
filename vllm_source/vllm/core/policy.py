from collections import deque
from typing import Deque, Dict, List
import gurobipy as gp
import time
import sys, os
import numpy as np
import pandas as pd
from vllm.sequence import SequenceGroup
import random

class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        #print('=====================')
        #print("metrics:", seq_group.metrics)
        return now - seq_group.metrics.arrival_time

class RandomPolicy(Policy):

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        seq_groups = list(seq_groups)
        random.shuffle(seq_groups)
        return deque(seq_groups)

class DeadlinePrioritizePolicy(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return seq_group.metrics.deadline

class BiddingPolicy(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        remaining_tokens = seq_group.metrics.tokens - seq_group.metrics.processed_token
        remaining_iterations = (seq_group.metrics.deadline - now) / 0.02  
        return remaining_tokens / max(remaining_iterations, 1)

class OfflineSolverPolicy(Policy):
    def __init__(self, planning_window_size: int = 1000, max_batch_size: int = 16, reserve: int = 0):
        self.planning_window_size = planning_window_size
        self.max_batch_size = max_batch_size
        self.reserve = reserve
        self.solved_priorities: Dict[int, float] = {}
        self.now = time.time()
        self.call_count = 0
    
    def solve_and_assign_priorities(self, now: float, seq_groups: Deque[SequenceGroup]):
        """Solve the optimization problem and assign priorities based on the solution."""
        all_requests = list(seq_groups)

        N = len(all_requests)
        if N == 0:
            return
        T = self.planning_window_size

        options = {
            "WLSACCESSID": os.getenv('WLSACCESSID'),
            "WLSSECRET": os.getenv("WLSSECRET"),
            "LICENSEID": int(os.getenv("LICENSEID"))
        }
        with gp.Env(params=options) as env, gp.Model(env=env) as model:
            # Create a new Gurobi model
            #model = gp.Model("OnlineScheduler")
            model.Params.LogToConsole = 0
            model.setParam('LogFile', 'online.solver')
            model.Params.Presolve = -1

            # Decision variables
            x = model.addVars(N, T, vtype=gp.GRB.BINARY, name="x")
            finished = model.addVars(N, vtype=gp.GRB.BINARY, name="finished")
            b = self.max_batch_size

            # Objective: maximize the number of completed sequences plus sum of request processing
            objective = gp.quicksum(finished[i] for i in range(N)) + gp.quicksum(
                gp.quicksum(x[i, t] for t in range(T)) for i in range(N))
            model.setObjective(objective, gp.GRB.MAXIMIZE)

            # Constraints
            inference_time = 0.4
            for i, req in enumerate(all_requests):
                if isinstance(req, SequenceGroup):
                    time_to_deadline = int((req.metrics.deadline - now)//inference_time)
                    T_req = min(T, int(time_to_deadline))
                    model.addConstr(
                        gp.quicksum(x[i, t] for t in range(T_req)) >= (req.metrics.tokens - req.metrics.processed_token) * finished[i],
                        f"Completion_{i}",
                    )

            # Batch size constraints
            model.addConstr(gp.quicksum(x[i, 0] for i in range(N)) <= b)
            for t in range(1, self.planning_window_size):
                model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= b - self.reserve)

            # Solve the model
            model.optimize()

            if model.status == gp.GRB.INFEASIBLE:
                print("Model is infeasible. Computing IIS...")
                model.computeIIS()
                model.write("infeasible_model.ilp")
            elif model.status == gp.GRB.OPTIMAL:
                model.write("saved_model.lp")
                model.write("saved_solution.sol")
            else:
                print(f"Optimization was unsuccessful. Status code: {model.status}")

    def get_priority(self, x, seq_group: SequenceGroup) -> float:
        """Return the precomputed priority from the Gurobi solver results."""
        #if not seq_group.is_prefill():
        #    seq_group.get_last_latency(self.now)
        #    if seq_group.metrics.processed_token > 0:
        #        print('========================')
        #        print("metrics ", seq_group.metrics)
        # Assign priorities based on solver results
        var = x[int(seq_group.request_id)*self.planning_window_size+self.call_count]
        if hasattr(var, 'X'):
            return var.X
        if hasattr(var, "Xn"):
            return var.Xn
        return 0

    def sort_by_priority(self, now: float, seq_groups: Deque[SequenceGroup]) -> Deque[SequenceGroup]:
        """Solve the optimization problem and sort the sequence groups by the computed priorities."""
        if self.call_count == 0:
            self.solve_and_assign_priorities(now, seq_groups)
        model = gp.read("saved_model.lp")
        model.read("saved_solution.sol")
        model.update()
        x = model.getVars()
        priorities = deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(x=x, seq_group=seq_group),
                reverse=True,
            ))
        self.call_count += 1
        return priorities
    

class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'random': RandomPolicy, 'deadline': DeadlinePrioritizePolicy, 'bidding': BiddingPolicy, 'offline': OfflineSolverPolicy}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

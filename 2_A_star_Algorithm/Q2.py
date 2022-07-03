from queue import PriorityQueue

import numpy as np


class State:
    def __init__(self, input_state, g_cost):
        self.state = input_state
        self.g_cost = g_cost
        self.h_cost = 0
        self.f_cost = self.g_cost + self.h_cost

    def calculate_f_cost(self, M, routes):  # 找下山方向的最短路径
        if State == 1:  # 已经到山下了
            return

        shortest_route = np.iinfo(np.int8).max
        for i in range(M):
            if routes[i, 0] == self.state and routes[i, 2] < shortest_route:
                shortest_route = routes[i, 2]
        self.h_cost = shortest_route
        self.f_cost = self.g_cost + self.h_cost

    def find_next(self, M, routes):  # 下一条可行的路(下个地标)
        next_routes = []
        for i in range(M):
            if routes[i, 0] == self.state:
                next_routes.append(routes[i, 1])
        return next_routes

    def __lt__(self, other):
        return self.f_cost < other.f_cost


def A_star(N, M, K, routes):
    # 建立关于各地标的邻接矩阵
    adjacency_mat = np.zeros((N + 1, N + 1), dtype=int)
    adjacency_mat[:, :] = np.iinfo(np.int8).max
    for i in range(M):
        adjacency_mat[routes[i, 0], routes[i, 1]] = routes[i, 2]

    priority_queue = PriorityQueue()
    shortest_cost = []
    present_state = State(N, 0)
    priority_queue.put(present_state)

    while not priority_queue.empty() and len(shortest_cost) < K:
        present_state = priority_queue.get()
        if present_state.state == 1:  # 到底了
            shortest_cost.append(present_state.g_cost)
        next_routes = present_state.find_next(M, routes)
        for i in range(len(next_routes)):
            next_g_cost = present_state.g_cost + adjacency_mat[present_state.state, next_routes[i]]  # 更新 g_cost
            next_state = State(next_routes[i], next_g_cost)
            next_state.calculate_f_cost(M, routes)
            priority_queue.put(next_state)

    if priority_queue.empty():  # 如果不存在就输出−1
        for _ in range(K - len(shortest_cost)):
            shortest_cost.append(-1)
    for cost in shortest_cost:
        print(cost)


if __name__ == "__main__":
    N, M, K = tuple(map(int, input().split(" ")))
    routes = np.zeros((M, 3), dtype=int)  # 各地标以及之间的路径
    for i in range(M):
        routes[i] = np.array(input().split(" "), dtype=int)

    A_star(N=N, M=M, K=K, routes=routes)

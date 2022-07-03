from queue import PriorityQueue

import numpy as np

# 终止状态
ultimate_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]], dtype=int)


class State:
    def __init__(self, state_mat, g_cost):
        self.state_mat = state_mat  # 状态矩阵
        self.g_cost = g_cost
        self.h_cost = self.calculate_h_cost()
        self.f_cost = self.g_cost + self.h_cost
        self.blank_loc = (np.where(self.state_mat == 0)[0][0], np.where(self.state_mat == 0)[1][0])  # 空白格所在坐标

    # 计算曼哈顿距离
    def calculate_h_cost(self):
        h_cost = 0
        for i in self.state_mat.flatten():
            h_cost += abs(np.where(self.state_mat == i)[0][0] - np.where(ultimate_state == i)[0][0]) + \
                      abs(np.where(self.state_mat == i)[1][0] - np.where(ultimate_state == i)[1][0])
        return h_cost

    def find_next(self):
        next_states = []
        # 空白格左移
        if self.blank_loc[1] > 0:
            next_state = self.state_mat.copy()
            next_state[self.blank_loc[0]][self.blank_loc[1]] = next_state[self.blank_loc[0]][self.blank_loc[1] - 1]
            next_state[self.blank_loc[0]][self.blank_loc[1] - 1] = 0
            next_states.append(next_state)

        # 空白格右移
        if self.blank_loc[1] < 2:
            next_state = self.state_mat.copy()
            next_state[self.blank_loc[0]][self.blank_loc[1]] = next_state[self.blank_loc[0]][self.blank_loc[1] + 1]
            next_state[self.blank_loc[0]][self.blank_loc[1] + 1] = 0
            next_states.append(next_state)

        # 空白格上移
        if self.blank_loc[0] > 0:
            next_state = self.state_mat.copy()
            next_state[self.blank_loc[0]][self.blank_loc[1]] = next_state[self.blank_loc[0] - 1][self.blank_loc[1]]
            next_state[self.blank_loc[0] - 1][self.blank_loc[1]] = 0
            next_states.append(next_state)

        # 空白格下移
        if self.blank_loc[0] < 2:
            next_state = self.state_mat.copy()
            next_state[self.blank_loc[0]][self.blank_loc[1]] = next_state[self.blank_loc[0] + 1][self.blank_loc[1]]
            next_state[self.blank_loc[0] + 1][self.blank_loc[1]] = 0
            next_states.append(next_state)
        return next_states

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return (self.state_mat == other.state_mat).all()


def A_star(initial_state):
    priority_queue = PriorityQueue()
    visited = []  # 已经访问过的状态
    state = State(initial_state, 0)
    priority_queue.put(state)

    n = 0  # 统计步数
    while not priority_queue.empty():
        present_state = priority_queue.get()  # 取出代价最小的可行状态
        if (present_state.state_mat == ultimate_state).all():  # 如果到了最终状态
            break
        n += 1
        visited.append(str(present_state.state_mat.flatten()))

        # 寻找下一步的可行状态
        next_states = present_state.find_next()
        for i in range(len(next_states)):
            if str(next_states[i].flatten()) in visited:  # 跳过已访问过的
                continue
            state_next = State(next_states[i], present_state.g_cost + 1)
            priority_queue.put(state_next)
    print(n)


if __name__ == "__main__":
    A_star(np.array(list(input()), dtype=int).reshape(3, 3))

import numpy as np


size = 5

start = (4, 4)
goal = (0, 0)

moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def is_valid(x, y):
    return 0 <= x < size and 0 <= y < size


def initialize_matrices(size):
    R = np.full((size * size, size * size), -1)
    Q = np.zeros((size * size, size * size))

    for x in range(size):
        for y in range(size):
            current_state = x * size + y
            for move in moves:
                nx, ny = x + move[0], y + move[1]
                if is_valid(nx, ny):
                    next_state = nx * size + ny
                    if (nx, ny) == goal:
                        R[current_state, next_state] = 100
                    else:
                        R[current_state, next_state] = 0
    R[0, 0] = 100
    return R, Q


def q_learning(R, Q, episodes=1000, gamma=0.8, alpha=0.1):
    for episode in range(episodes):
        state = start[0] * size + start[1]
        while state != goal[0] * size + goal[1]:
            possible_actions = np.where(R[state, :] >= 0)[0]
            if len(possible_actions) == 0:
                break

            action = np.random.choice(possible_actions)

            reward = R[state, action]
            next_state = action
            Q[state, action] = reward + gamma * np.max(Q[next_state, :])
            # Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
    return Q


def show_path(Q):
    state = start[0] * size + start[1]
    path = [state]
    while state != goal[0] * size + goal[1]:
        action = np.argmax(Q[state, :])
        state = action
        path.append(state)
    return path


def main():
    R, Q = initialize_matrices(size)
    Q = q_learning(R, Q)

    np.set_printoptions(linewidth=np.inf, precision=2)
    print(Q, "\n")

    path = show_path(Q)
    for state in path:
        x, y = divmod(state, size)
        print(f"({x}, {y})", end=" -> ")
    print("Goal")


if __name__ == "__main__":
    main()

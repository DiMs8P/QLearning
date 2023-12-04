import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0", render_mode='rgb_array')
env.metadata['render_fps'] = 150

env.reset()

# его основная цель — избежать больших изменений в одном обновлении, поэтому вместо того, чтобы лететь в цель,
# мы медленно приближаемся к ней.
LEARNING_RATE = 0.1

# Обесценивание будущей награды
DISCOUNT = 0.95  # How important future values over current values(rewarn fi)
EPISODES = 2_001

SHOW_EVERY = 500

# Наазчальное значение epsilon Так как мы всегда предпринимаем действия, которые считаем лучшими, то мы застрянем с
# первой найденной наградой, всегда возвращаясь к ней Нам нужно убедиться, что мы достаточно изучили наш мир (это
# удивительно трудная задача). Вот где вступает в игру ε. ε в жадном алгоритме означает, что мы должны действовать
# жадно, НО делать случайные действия в процентном соотношении ε по времени, таким образом, при бесконечном
# количестве попыток мы должны исследовать все состояния.
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

# Насколько будет изменяться epsilon на каждом эпизоде
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))


print("START")
for episode in range(EPISODES):
    episode_reward = 0
    reset_state, _ = env.reset()
    discrete_state = get_discrete_state(reset_state)
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(f"{episode}")
    else:
        render = False

    while not done:

        # Учёт epsilon
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Отрисовка эпизода
        if render:
            image = env.render()
            cv2.imshow("image", image)
            cv2.waitKey(1)

        done = terminated or truncated
        new_discrete_state = get_discrete_state(observation)
        if not done:
            # наилучшая награда для будущего шага (учитывается при изменении награды на текущем шаге, обесценивается
            # параметром DISCOUNT)
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            # Формула подсчёта нового Q значения для текущего состояния
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif observation[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            print(q_table)
            # Значение Q для конечных состояний равно нулю, мы не можем предпринимать никаких действий в конечных
            # состояниях, поэтому мы считаем значение для всех действий в этом состоянии равным нулю.
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

    episode_reward = 0

print("END")
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')

plt.legend(loc=4)
plt.show()

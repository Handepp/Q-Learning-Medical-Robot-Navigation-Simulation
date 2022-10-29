
from Env_QL import Environment
from agent_QL import QLearningTable
import time


# -----function for the training loop-----
def train():
    steps = []               # record steps of each episode
    Q_sum = []               # summed Q_value for all episode
    success_times = 0        # success times of all episodes
    t = 0                    # record each episode running time
    t_sum = []               # record the sum time of training loop

    for episode in range(800):
        start = time.time()  # Start time of each episode
        s = env.reset()      # initial observation
        i = 0                # update number of steps for each episode
        Q = 0                # update the Q for each episode

        while True:
            env.render()                                 # Refresh environment
            a = RL.choose_action(str(s))                 # choose action based on observation
            s_, r, done = env.step(a)                    # take an action and get the next observation and reward
            print(s,a,r)
            Q += RL.update_table(str(s), a, r, str(s_))  # learn from this transition and calculate the Q value
            s = s_                                       # swap the current and next observation
            i += 1                                       # steps of each episode plus one

            if env.next_state == [650, 550]:             # when agent reach the goal coordinate
                success_times += 1                       # when the agent reach the goal +1

            # when agent reached the obstacle or goal
            if done:
                steps += [i]                             # steps of each episode
                Q_sum += [Q]                             # Q_sum of each episode
                end = time.time()                        # end time of each episode
                t_sum.append(t)                          # append each time of episode into a list
                t += end - start                         # time of each episode
                break

    print('Durasi pembelajaran:', t)                            # print the simulation time of the algorithm
    print('Jumlah sukses:', success_times)               # print the success times
    success_rate = success_times / 800
    print('Tingkat kesuksesan: {:.2%}'.format(success_rate))   # show the success rate

    env.final()                           # show the final route

    RL.print_q_table()                    # show the Q-table

    RL.plot_results(steps, Q_sum, t_sum)  # plot the Q_sum and steps over episodes


if __name__ == "__main__":
    # call for the environment
    env = Environment()
    # input the actions and states to call for the main algorithm
    RL = QLearningTable(actions=[0, 1, 2, 3],
                        states=['[50.0, 50.0]', '[50.0, 150.0]', '[50.0, 250.0]', '[50.0, 350.0]','[50.0, 450.0]','[50.0, 550.0]','[50.0, 650.0]',
                                '[150.0, 50.0]', '[150.0, 150.0]', '[150.0, 250.0]', '[150.0, 350.0]','[150.0, 450.0]', '[150.0, 550.0]','[150.0, 650.0]',
                                '[250.0, 50.0]', '[250.0, 150.0]', '[250.0, 250.0]', '[250.0, 350.0]','[250.0, 450.0]', '[250.0, 550.0]','[250.0, 650.0]',
                                '[350.0, 50.0]', '[350.0, 150.0]', '[350.0, 250.0]', '[350.0, 350.0]','[350.0, 450.0]', '[350.0, 550.0]','[350.0, 650.0]',
                                '[450.0, 50.0]', '[450.0, 150.0]', '[450.0, 250.0]', '[450.0, 350.0]','[450.0, 450.0]', '[450.0, 550.0]','[450.0, 650.0]',
                                '[550.0, 50.0]', '[550.0, 150.0]', '[550.0, 250.0]', '[550.0, 350.0]','[550.0, 450.0]', '[550.0, 550.0]','[550.0, 650.0]',
                                '[650.0, 50.0]', '[650.0, 150.0]', '[650.0, 250.0]', '[650.0, 350.0]','[650.0, 450.0]', '[650.0, 550.0]','[650.0, 650.0]'],)
    env.after(800, train)
    env.mainloop()

from DQN import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent((6, 7), 7, 50000, 1000, .005, .99, .0, 0.01)
    agent1.load("connect4-cnn-agent1-model1.pt", "connect4-cnn-agent1-model2.pt")
    
    state = env.reset(False)
    done = False
    while not done:
        action = agent1.choose_action(state, env.mask)
        state, _, _, done = env.step(action, 1.)
        if done:
            break
        print(env)
        action = int(input("Position: "))
        state, _, _, done = env.step(action, -1.)
        if done:
            break
    print(env)
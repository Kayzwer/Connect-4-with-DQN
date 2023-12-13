from DQN import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent((6, 7), 7, 10, 1, .01, .99, 1.0, 0.01, 10, 0.01)
    agent1.load("weights\connect4-cnn-agent1-model1.pt", "weights\connect4-cnn-agent1-model2.pt")
    
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
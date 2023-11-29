from SemiGradientSARSA import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent2 = Agent(0.01, 0.99, .0, 7)
    agent2.load("connect4-agent2.pt")
    
    state = env.reset(False)
    done = False
    while not done:
        print(env)
        action = int(input("Position: "))
        state, _, _, done = env.step(action, 1.)
        if done:
            break
        action = agent2.choose_action(state, env.mask)
        state, _, _, done = env.step(action, -1.)
        if done:
            break
    print(env)
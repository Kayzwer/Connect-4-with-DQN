from DQN import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent((6, 7), 7, 100000, 128, .0025, .99, 1.0, 0.05, 100000, 0.01)
    agent2 = Agent((6, 7), 7, 100000, 128, .0025, .99, 1.0, 0.05, 100000, 0.01)
    episodes = 5000000
    iteration_to_save = 500
    iteration_to_log_game = 1000

    for episode in range(episodes):
        agent1_total_loss = 0.
        agent2_total_loss = 0.
        done = False

        state_agent1 = env.reset((episode + 1) % iteration_to_log_game == 0)
        action_agent1 = agent1.choose_action(state_agent1, env.mask)

        state_agent2, reward1, reward2, done = env.step(action_agent1, 1.)
        action_agent2 = agent2.choose_action(state_agent2, env.mask)

        while not done:
            next_state_agent1, reward1, reward2, done = env.step(
                action_agent2, -1.)
            if done:
                agent2_total_loss += agent2.update(state_agent2, action_agent2,
                                                   reward2, next_state_agent1,
                                                   done)
                agent1_total_loss += agent1.update(state_agent1, action_agent1,
                                                   reward1, next_state_agent1,
                                                   done)
                break
            agent1_total_loss += agent1.update(state_agent1, action_agent1,
                                               reward1, next_state_agent1, done)

            state_agent1 = next_state_agent1
            action_agent1 = agent1.choose_action(state_agent1, env.mask)

            next_state_agent2, reward1, reward2, done = env.step(
                action_agent1, 1.)
            if done:
                agent1_total_loss += agent1.update(state_agent1, action_agent1,
                                                   reward1, next_state_agent2,
                                                   done)
                agent2_total_loss += agent2.update(state_agent2, action_agent2,
                                                   reward2, next_state_agent2,
                                                   done)
                break
            agent2_total_loss += agent2.update(state_agent2, action_agent2,
                                               reward2, next_state_agent2,
                                               done)
            state_agent2 = next_state_agent2
            action_agent2 = agent2.choose_action(state_agent2, env.mask)
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        if (episode + 1) % iteration_to_save == 0:
            print("Checkpoint")
            agent1.save("connect4-cnn-agent1-model1.pt", "connect4-cnn-agent1-model2.pt")
            agent2.save("connect4-cnn-agent2-model1.pt", "connect4-cnn-agent2-model2.pt")
        winner = "Draw"
        if env.winner == 1.:
            winner = "Agent1"
        elif env.winner == -1.:
            winner = "Agent2"
        env.save_game_state_log(f"./game_state_log/game_{episode + 1}.txt")
        print(f"Episode: {episode + 1}, Winner: {winner}, Agent1 Loss: "
              f"{agent1_total_loss:.3f}, Agent2 Loss: {agent2_total_loss:.3f}")
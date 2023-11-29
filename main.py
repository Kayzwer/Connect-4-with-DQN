from SemiGradientSARSA import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent(.005, .99, .05, 7)
    agent1.load(r"connect4-agent1.pt")
    agent2 = Agent(.005, .99, .05, 7)
    agent2.load(r"connect4-agent2.pt")
    episodes = 50000000
    iteration_to_save = 5000
    iteration_to_log_game = 1000
    iteration_to_update_target_network = 50000

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
                agent2_total_loss += agent2.update(done, state_agent2,
                                                   action_agent2, reward2)
                agent1_total_loss += agent1.update(done, state_agent1,
                                                   action_agent1, reward1)
                break
            agent1_total_loss += agent1.update(
                done, state_agent1, action_agent1, reward1, next_state_agent1,
                agent1.choose_action(next_state_agent1, env.mask))

            state_agent1 = next_state_agent1
            action_agent1 = agent1.choose_action(state_agent1, env.mask)

            next_state_agent2, reward1, reward2, done = env.step(
                action_agent1, 1.)
            if done:
                agent1_total_loss += agent1.update(done, state_agent1,
                                                   action_agent1, reward1)
                agent2_total_loss += agent2.update(done, state_agent2,
                                                   action_agent2, reward2)
                break
            agent2_total_loss += agent2.update(
                done, state_agent2, action_agent2, reward2, next_state_agent2,
                agent2.choose_action(next_state_agent2, env.mask))
            state_agent2 = next_state_agent2
            action_agent2 = agent2.choose_action(state_agent2, env.mask)
        if (episode + 1) % iteration_to_update_target_network == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print("Target Network Updated")
        if (episode + 1) % iteration_to_save == 0:
            print("Checkpoint")
            agent1.save("connect4-agent1.pt")
            agent2.save("connect4-agent2.pt")
        winner = "Draw"
        if env.winner == 1.:
            winner = "Agent1"
        elif env.winner == -1.:
            winner = "Agent2"
        env.save_game_state_log(f"./game_state_log/game_{episode + 1}.txt")
        print(f"Episode: {episode + 1}, Winner: {winner}, Agent1 Loss: "
              f"{agent1_total_loss:.3f}, Agent2 Loss: {agent2_total_loss:.3f}")

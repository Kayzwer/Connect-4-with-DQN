from A2C import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent(.000025, .0001, .99, 0.1, 100000, 128, (6, 7), 0.01)
    agent2 = Agent(.000025, .0001, .99, 0.1, 100000, 128, (6, 7), 0.01)
    episodes = 5000000
    iteration_to_save = 500
    iteration_to_log_game = 5000

    for episode in range(episodes):
        agent1_total_value_loss = 0.
        agent1_total_policy_loss = 0.
        agent2_total_value_loss = 0.
        agent2_total_policy_loss = 0.
        done = False

        state_agent1 = env.reset((episode + 1) % iteration_to_log_game == 0)
        action_agent1 = agent1.choose_action(state_agent1, env.mask)

        state_agent2, reward1, reward2, done = env.step(action_agent1, 1.)
        action_agent2 = agent2.choose_action(state_agent2, env.mask)

        while not done:
            next_state_agent1, reward1, reward2, done = env.step(
                action_agent2, -1.)
            if done:
                agent2_value_loss, agent2_policy_loss = agent2.update(
                    state_agent2, action_agent2, reward2, next_state_agent1,
                    done)
                agent1_value_loss, agent1_policy_loss = agent1.update(
                    state_agent1, action_agent1, reward1, next_state_agent1,
                    done)

                agent2_total_value_loss += agent2_value_loss
                agent2_total_policy_loss += agent2_policy_loss
                agent1_total_value_loss += agent1_value_loss
                agent1_total_policy_loss += agent1_policy_loss
                break
            agent1_value_loss, agent1_policy_loss = agent1.update(
                state_agent1, action_agent1, reward1, next_state_agent1, done)
            agent1_total_value_loss += agent1_value_loss
            agent1_total_policy_loss += agent1_policy_loss

            state_agent1 = next_state_agent1
            action_agent1 = agent1.choose_action(state_agent1, env.mask)

            next_state_agent2, reward1, reward2, done = env.step(
                action_agent1, 1.)
            if done:
                agent1_value_loss, agent1_policy_loss = agent1.update(
                    state_agent1, action_agent1, reward1, next_state_agent2,
                    done)
                agent2_value_loss, agent2_policy_loss = agent2.update(
                    state_agent2, action_agent2, reward2, next_state_agent2,
                    done)

                agent1_total_value_loss += agent1_value_loss
                agent1_total_policy_loss += agent1_policy_loss
                agent2_total_value_loss += agent2_value_loss
                agent2_total_policy_loss += agent2_policy_loss
                break
            agent2_value_loss, agent2_policy_loss = agent2.update(
                state_agent2, action_agent2, reward2, next_state_agent2, done)
            agent2_total_value_loss += agent2_value_loss
            agent2_total_policy_loss += agent2_policy_loss

            state_agent2 = next_state_agent2
            action_agent2 = agent2.choose_action(state_agent2, env.mask)
        if (episode + 1) % iteration_to_save == 0:
            print("Checkpoint")
            agent1.save("connect4-cnn-actor-network-agent1.pt",
                        "connect4-cnn-critic-network-agent1.pt",
                        "connect4-cnn-target-critic-network-agent1.pt")
            agent2.save("connect4-cnn-actor-network-agent2.pt",
                        "connect4-cnn-critic-network-agent2.pt",
                        "connect4-cnn-target-critic-network-agent2.pt")
        winner = "Draw"
        if env.winner == 1.:
            winner = "Agent1"
        elif env.winner == -1.:
            winner = "Agent2"
        env.save_game_state_log(f"./game_state_log/game_{episode + 1}.txt")
        print(f"Episode: {episode + 1}, Winner: {winner}, Agent1 Value Loss: "
              f"{agent1_total_value_loss:.3f}, Agent1 Policy Loss: "
              f"{agent1_total_policy_loss:.3f}, Agent2 Value Loss: "
              f"{agent2_total_value_loss:.3f}, Agent2 Policy Loss: "
              f"{agent2_total_policy_loss:.3f}")

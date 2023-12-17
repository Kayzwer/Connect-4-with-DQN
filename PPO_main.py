from PPO import Agent
from Connect4 import Connect4


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent(.000025, .0001, 16, .99, .97, .2, .05, 0.1, 32, 8)
    agent2 = Agent(.000025, .0001, 16, .99, .97, .2, .05, 0.1, 32, 8)
    episodes = 5000000
    iteration_to_save = 500
    iteration_to_log_game = 5000

    for episode in range(episodes):
        done = False

        state_agent1 = env.reset((episode + 1) % iteration_to_log_game == 0)
        action_agent1, action_prob_agent1, state_value_agent1 = \
            agent1.choose_action(state_agent1, env.mask)

        state_agent2, reward1, reward2, done = env.step(action_agent1, 1.)
        action_agent2, action_prob_agent2, state_value_agent2 = \
            agent2.choose_action(state_agent2, env.mask)

        while not done:
            next_state_agent1, reward1, reward2, done = env.step(action_agent2,
                                                                 -1.)
            agent1.store(episode, state_agent1, state_value_agent1,
                         action_agent1, action_prob_agent1, reward1, done)
            if done:
                agent2.store(episode, state_agent2, state_value_agent2,
                             action_agent2, action_prob_agent2, reward2, done)
                break

            state_agent1 = next_state_agent1
            action_agent1, action_prob_agent1, state_value_agent1 = \
                agent1.choose_action(state_agent1, env.mask)

            next_state_agent2, reward1, reward2, done = env.step(
                action_agent1, 1.)
            agent2.store(episode, state_agent2, state_value_agent2,
                         action_agent2, action_prob_agent2, reward2, done)
            if done:
                agent1.store(episode, state_agent1, state_value_agent1,
                             action_agent1, action_prob_agent1, reward1, done)
                break

            state_agent2 = next_state_agent2
            action_agent2, action_prob_agent2, state_value_agent2 = \
                agent2.choose_action(state_agent2, env.mask)

        agent1_actor_loss, agent1_critic_loss = agent1.update(episode)
        agent2_actor_loss, agent2_critic_loss = agent2.update(episode)
        if (episode + 1) % iteration_to_save == 0:
            print("Checkpoint")
            agent1.save("connect4-cnn-actor-network-agent1.pt",
                        "connect4-cnn-critic-network-agent1.pt")
            agent2.save("connect4-cnn-actor-network-agent2.pt",
                        "connect4-cnn-critic-network-agent2.pt")
        winner = "Draw"
        if env.winner == 1.:
            winner = "Agent1"
        elif env.winner == -1.:
            winner = "Agent2"
        env.save_game_state_log(f"./game_state_log/game_{episode + 1}.txt")
        print(f"Episode: {episode + 1}, Winner: {winner}, Agent1 Actor Loss: "
              f"{agent1_actor_loss:.3f}, Agent1 Critic Loss: "
              f"{agent1_critic_loss:.3f}, Agent2 Actor Loss: "
              f"{agent2_actor_loss:.3f}, Agent2 Critic Loss: "
              f"{agent2_critic_loss:.3f}")

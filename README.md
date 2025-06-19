# How can we achieve convergence by only priortizing what matters the most in Q-learning?
AI Algorithms sometimes need a small human touch to see a big difference.Solved one of the toughest issues in reinforcement learning’s classic FrozenLake-8x8 environment — without reward shaping or deep learning!I derived a new method for convergence of Q-learning in Frozen-lake environment-8x8 of Gymnasium just by removing the noise learned and changing the exploration strategy from linear to sigmoid.It is simple yet clever method. It is like attention is all you need prioritizing what matters the most.

# Problem:
Q-learning becomes hard to converge when it comes to only rewarding our agent(software) when it achieves its goal specially in gymnasium environment initially developed by OpenAI but now maintained by Farama Foundation. In Frozen-Lake environment they have assigned 1 reward when the goal is achieved and 0 rest all of the time whether it makes a mistake or just a normal step.  The problem here is the Q values are too sensitive towards the steps it makes and the Q-values influenced(learned) by achieving the goal in any episode could change if multiple unachieved episodes are done because of the low reward in terms of numbers. One solution could be just change the reward-penalty model i.e. let's say assign -10 reward when it makes mistakes and +10 reward when it achieves a goal boom our algorithm converges in few episodes and achieves optimal values. But then this the problem with tweaked reward modeling is not realistic because of the randomness in the real world problems. We don't know which action must be penalized because of the large action and state space in real world scenarios.

Fact: Achieving goal is tough but making mistakes is very easy and frequent.


# Solution: Q-table rollback mechanism + a smooth sigmoid epsilon decay:
 1. Innovation:1- I though why to learn the Q-values when it didn't achieve the goal? Why to learn those paths which are just loops?Why to learn those paths which lead to hole ? Let's unlearn them and give priority to success only.
 2. Innovation:2- As we are not learning neither normal steps nor mistakes so it is very important for us to explore more to reach the goal so I kept the exploration high initially up to a certain steps(5000 steps in my case) and then I decreased it sharply to not explore but to follow the learned paths more.
    This exploration-exploitation behavior can be achieved by a sigmoid function.
# Result:
 1. Almost guaranteed convergence
 2. Reduces number of training episodes sharply. 



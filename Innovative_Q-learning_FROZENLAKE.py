import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
print(device)



def run(episodes, render=True):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False, render_mode='rgb_array' if render else None)
    q= np.zeros((env.observation_space.n, env.action_space.n)) # 64x4
    print(env.observation_space, env.action_space)
    
    print("Transition_probability, ", env.unwrapped.P[15][2]) 
    '''
        env.P[state][action] ,
        this gives an output: [(transition probability, next state, reward, Is terminal state?)] this depends upon whether=> is_slippery=False or True!, if it is True then our environmnet is stochastic otherwise deterministic means only one outcome
    '''

    lr=0.2
    df=0.99
    
    epsilon=1
    rewards=[]
    reward_per_100=[]
    

    for i in range(1,episodes+1):
        state= env.reset()[0] # states: 0 to 63
        terminated=False # true when we fall in hole or in the goal
        truncated=False # if actions>200

        q_old=q
        steps=0
        while (not terminated and not truncated):
            if random.random()<epsilon:
                action=env.action_space.sample() #actions: 0=left, 1= down, 2= right, 3=up
                
            else:
                action=np.argmax(q[state,:])
        
            new_state,reward,terminated,truncated,prob=env.step(action)
            #print(prob) this is the transition prob for that action
            
            
            q[state,action]=q[state,action]+lr*(reward+df*np.max(q[new_state,:])-q[state,action])
            state=new_state
            
            steps+=1
            if (steps==200 or truncated) and not terminated:
                q=q_old.copy()
                break
            
           
            
            
        rewards.append(reward)
        epsilon = max(0.2, 1 / (1 + np.exp(10 * (i - 5000))))
        

        if reward == 1:
            print(f"🎯 Goal reached at episode {i}")

        if i%100==0:
            reward_per_100.append(sum(rewards))
            rewards=[]
        
    env.close()
   
    
    
    x_axis=[k for k in range(len(reward_per_100))]
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis,reward_per_100)
    plt.grid()
    plt.show()

if __name__== '__main__':
    run(10000)
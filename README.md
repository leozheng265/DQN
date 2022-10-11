# Reinforcement learning with Pong

## Introduction

This project implements the deep Q-learning algorithm to train a model to play the game Pong without any human knowledge of the game. OpenAI gym is used to run and interact with the Pong Atari game.

## Testing and Training

A pre-trained model, which is able to play Pong but performs very poorly, is used as a starting point. This model achieves an average score of 7. To run this model:

```bash
python3 test_dqn_pong.py model_pretrained.pth
```

To train this model with deep Q-learning, run the command:

```bash
python3 run_dqn_pong.py model_pretrained.pth
```

This process will take several hours depending on performance of GPU.

A well-trained model is also provided. This model achieves an average score of 21. To run this model:

```bash
python3 test_dqn_pong.py model_trained.pth
```

## Demo video of running the models:

https://youtu.be/9BRiTCtYk2M
